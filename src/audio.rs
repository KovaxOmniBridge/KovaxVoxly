use crate::engine;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use std::sync::mpsc;
use std::thread;

pub fn start_bridge() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let mic = host.default_input_device().expect("Mic não encontrado");
    let out = host.default_output_device().expect("Saída não encontrada");

    let config: cpal::StreamConfig = mic.default_input_config()?.into();
    let native_sr = config.sample_rate.0;

    let mut denoiser = engine::Denoiser::new(native_sr)?;
    let df_size = denoiser.frame_size();
    let mut df_buffer = Vec::with_capacity(df_size);

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let (tx_whisper, mut rx_whisper) = tokio::sync::mpsc::unbounded_channel::<(usize, Vec<f32>)>();

    // --- THREAD DE INTELIGÊNCIA ---
    thread::spawn(move || {
        let mut vad = engine::Vad::new("silero_vad.onnx").expect("Erro VAD");
        let mut diarizer = engine::Diarizer::new("segmentation-3.0.onnx").expect("Erro Diarizer");

        let mut pre_roll = engine::buffers::PreRollBuffer::new(8000);
        let mut recording_buffer = Vec::new();
        let mut is_recording = false;
        let mut silence_frames = 0;
        let mut current_speaker = 0;

        println!("[IA] Motores ativos. Fale no microfone...");

        while let Ok(samples) = rx.recv() {
            pre_roll.push_samples(&samples);

            let probability = vad.is_speech(samples.clone());

            // Ajuste de Sensibilidade: 0.30 é ideal para microfones comuns
            if probability > 0.30 {
                if !is_recording {
                    is_recording = true;
                    pre_roll.drain_to(&mut recording_buffer);
                }
                recording_buffer.extend(samples);
                silence_frames = 0;

                current_speaker = diarizer.identify(&recording_buffer);
                let bars = (probability * 20.0) as usize;
                let bar_str = "█".repeat(bars);
                let space_str = " ".repeat(20 - bars);

                use crossterm::{
                    cursor, execute,
                    style::{Color, Print, ResetColor, SetForegroundColor},
                    terminal,
                };
                let _ = execute!(
                    std::io::stdout(),
                    cursor::MoveToColumn(0),
                    terminal::Clear(terminal::ClearType::UntilNewLine),
                    SetForegroundColor(Color::Green),
                    Print("[VAD] "),
                    ResetColor,
                    Print(format!(
                        "[{}{}] {:.0}% | Locutor ",
                        bar_str,
                        space_str,
                        probability * 100.0
                    )),
                    SetForegroundColor(Color::Cyan),
                    Print(format!("{}", current_speaker)),
                    ResetColor,
                    Print(format!(
                        " | {:.1}s",
                        recording_buffer.len() as f32 / 16000.0
                    ))
                );
            } else if is_recording {
                recording_buffer.extend(samples);
                silence_frames += 1;

                use crossterm::{
                    cursor, execute,
                    style::{Color, Print, ResetColor, SetForegroundColor},
                    terminal,
                };
                let _ = execute!(
                    std::io::stdout(),
                    cursor::MoveToColumn(0),
                    terminal::Clear(terminal::ClearType::UntilNewLine),
                    SetForegroundColor(Color::Yellow),
                    Print("[VAD] "),
                    ResetColor,
                    Print(format!("Analisando silêncio... ({} / 30)", silence_frames))
                );

                if silence_frames > 30 {
                    println!(
                        "\n\x1b[32m[✓]\x1b[0m Frase finalizada. Limpando memória contextual..."
                    );
                    let _ = tx_whisper.send((current_speaker, recording_buffer.clone()));
                    recording_buffer.clear();
                    is_recording = false;

                    // Reseta os estados preditivos para a próxima frase não sofrer vazamento
                    vad.reset_states();
                    diarizer.reset_history();
                }
            }
        }
    });

    tokio::spawn(async move {
        // Tenta conectar ao microserviço Python de Transcrição.
        // Se a conexão cair, reconecta automaticamente sem perder o loop de ASR.
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

        println!("[WHISPER] Aguardando serviço Python local (127.0.0.1:8000)...");

        'whisper_bridge: loop {
            let mut stream = loop {
                match connect_async("ws://127.0.0.1:8000").await {
                    Ok((stream, _)) => {
                        println!("\x1b[32m[✓]\x1b[0m Conectado ao Whisper Python WS!");
                        break stream;
                    }
                    Err(_) => {
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    }
                }
            };

            while let Some((speaker_id, audio_data)) = rx_whisper.recv().await {
                // Converte o Vec<f32> em puro byte stream (Little Endian)
                let byte_data: Vec<u8> = audio_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

                if let Err(e) = stream
                    .send(Message::Binary(bytes::Bytes::from(byte_data)))
                    .await
                {
                    eprintln!(
                        "[WHISPER] Erro ao enviar áudio: {}. Tentando reconectar...",
                        e
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue 'whisper_bridge;
                }

                // Aguarda silenciosamente a resposta do Python em JSON
                match stream.next().await {
                    Some(Ok(Message::Text(txt))) => {
                        println!(
                            "\n\x1b[35m[TRANSCRIÇÃO -> Locutor {}]\x1b[0m {}",
                            speaker_id, txt
                        );
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        eprintln!("[WHISPER] Erro na resposta: {}. Tentando reconectar...", e);
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        continue 'whisper_bridge;
                    }
                    None => {
                        eprintln!("[WHISPER] Conexão encerrada pelo servidor. Reconectando...");
                        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                        continue 'whisper_bridge;
                    }
                }
            }

            break;
        }
    });

    let rb = HeapRb::<f32>::new(native_sr as usize * 2);
    let (mut prod, mut cons) = rb.split();

    let mut chunk_collector = Vec::with_capacity(512);
    let sample_step = native_sr as f32 / 16000.0;
    let mut current_pos = 0.0f32;
    let mut _debug_frames_captured = 0;

    let input_stream = mic.build_input_stream(
        &config,
        move |data: &[f32], _| {
            _debug_frames_captured += 1;
            if _debug_frames_captured % 100 == 0 {
                // Imprime a cada ~1-2 segundos apenas para provar que a thread sobreviveu
                // println!("[DEBUG] Capturados {} blocos do microfone", _debug_frames_captured);
            }

            for &s in data {
                df_buffer.push(s);

                if df_buffer.len() >= df_size {
                    // 1. Limpa o áudio (O que você ouve no fone)
                    denoiser.process_frame(&mut df_buffer);

                    for &clean_s in &df_buffer {
                        let _ = prod.push(clean_s);

                        // 2. Coleta para a IA (Resampling simplificado)
                        current_pos += 1.0;
                        if current_pos >= sample_step {
                            chunk_collector.push(clean_s);
                            current_pos = 0.0; // Reset simples

                            if chunk_collector.len() >= 512 {
                                // FORÇAR ENVIO: O Rio agora deságua na IA
                                if let Err(e) = tx.send(chunk_collector.clone()) {
                                    eprintln!("Erro ao enviar chunk para IA: {}", e);
                                }
                                chunk_collector.clear();
                            }
                        }
                    }
                    df_buffer.clear();
                }
            }
        },
        |e| eprintln!("Erro Mic: {}", e),
        None,
    )?;

    let output_stream = out.build_output_stream(
        &config,
        move |data: &mut [f32], _| {
            for s in data.iter_mut() {
                *s = cons.pop().unwrap_or(0.0);
            }
        },
        |e| eprintln!("Erro Saída: {}", e),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    println!("\n[VIVO] Voxly: {}Hz nativos.", native_sr);

    // Mantém os streams vivos sem depender de stdin.
    // Isso evita encerramento prematuro quando iniciado por launchers GUI.
    let (_shutdown_tx, shutdown_rx) = mpsc::channel::<()>();
    let _ = shutdown_rx.recv();

    Ok(())
}
