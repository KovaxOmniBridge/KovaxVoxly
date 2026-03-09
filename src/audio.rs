use crate::engine;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

pub fn start_bridge() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let mic = host.default_input_device().expect("Mic não encontrado");

    // Tenta encontrar a entrada do Virtual Cable que alimentará o Google Meet
    let out = host.output_devices()
        .expect("Falha ao listar saídas")
        .find(|d| d.name().unwrap_or_default().to_lowercase().contains("cable input"))
        .unwrap_or_else(|| {
            println!("\n[!] Aviso crítico: 'CABLE Input' não detectado.");
            println!("    Isso significa que o driver Virtual Cable não instalou corretamente ou precisa de boot.");
            println!("    Fallback: Usando saída de áudio padrão (caixas de som).\n");
            host.default_output_device().expect("Saída padrão não encontrada")
        });

    let config: cpal::StreamConfig = mic.default_input_config()?.into();
    let native_sr = config.sample_rate.0;
    let in_channels = config.channels as usize;

    let out_config: cpal::StreamConfig = out.default_output_config()?.into();
    let out_channels = out_config.channels as usize;

    let mut denoiser = engine::Denoiser::new(native_sr)?;
    let df_size = denoiser.frame_size();
    let mut df_buffer = Vec::with_capacity(df_size);

    // TX para o Microfone (Você)
    let (tx_mic, rx_mic) = mpsc::channel::<Vec<f32>>();
    // TX para o Loopback (Os outros na chamada)
    let (tx_sys, rx_sys) = mpsc::channel::<Vec<f32>>();

    // Novo Canal Unificado: ORIGEM (0 = Mic, 1 = Sis), IS_FINAL, Áudio f32
    let (tx_whisper, mut rx_whisper) =
        tokio::sync::mpsc::unbounded_channel::<(u8, bool, Vec<f32>)>();

    // Duplica o channel sender para uso nas duas threads
    let tx_whisper_mic = tx_whisper.clone();
    let tx_whisper_sys = tx_whisper.clone();

    // --- THREAD DE INTELIGÊNCIA ---
    thread::spawn(move || {
        let mut vad = engine::Vad::new("models/silero_vad.onnx").expect("Erro VAD");

        let mut pre_roll = engine::buffers::PreRollBuffer::new(8000);
        let mut recording_buffer = Vec::new();
        let mut is_recording = false;
        let mut silence_frames = 0;
        let mut last_partial_len = 0; // Para despachar transcrição parcial a cada X ms

        // Sensibilidade e limites de flush para não depender apenas de silêncio
        let vad_speech_threshold = 0.05f32;
        let max_chunk_samples = 16000usize * 8usize; // Limite longo fixo

        println!("[IA] Motores ativos. Fale no microfone...");

        while let Ok(samples) = rx.recv() {
            pre_roll.push_samples(&samples);

            let probability = vad.is_speech(samples.clone());

            // Sensibilidade mais permissiva para evitar perda de fala em microfones comuns
            if probability > vad_speech_threshold {
                if !is_recording {
                    is_recording = true;
                    pre_roll.drain_to(&mut recording_buffer);
                }
                recording_buffer.extend(samples);
                silence_frames = 0;

                // FASE 3 (STREAMING ASR): Disparar Pacotes Parciais a cada ~1.0 segundo (16000 samples).
                // Isso cria o efeito "Palavras Mágicas flutuantes" na interface (is_final = false).
                if recording_buffer.len() >= last_partial_len + 16000 {
                    let _ = tx_whisper.send((0, false, recording_buffer.clone())); // 0 = Microfone
                    last_partial_len = recording_buffer.len();
                }

                // Flush forçado absoluto (Segurança Limite de Memória e Demora)
                if recording_buffer.len() >= max_chunk_samples {
                    println!(
                        "\n\x1b[36m[WHISPER]\x1b[0m Flush absoluto por duração limite (~{:.1}s)",
                        recording_buffer.len() as f32 / 16000.0
                    );
                    let _ = tx_whisper.send((0, true, recording_buffer.clone())); // Força como FINAL
                    recording_buffer.clear();
                    last_partial_len = 0;
                    vad.reset_states();
                }

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
                        "[{}{}] {:.0}%",
                        bar_str,
                        space_str,
                        probability * 100.0
                    )),
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
                    Print(format!("Analisando silêncio... ({} / 20)", silence_frames))
                );

                // Usa 20 quadros de silêncio para não "cortar" palavras como T-estando
                // antes da pessoa terminar de emitir o som da consoante inicial.
                if silence_frames > 20 {
                    // CORTA O ÁUDIO E DECLARA FRASE FINAL (is_final = true)
                    if recording_buffer.len() > 16000 {
                        println!(
                            "\n\x1b[32m[✓]\x1b[0m Silêncio Detectado. Fechando frase e acionando NLLB Trnslator..."
                        );
                        let _ = tx_whisper.send((0, true, recording_buffer.clone()));
                        // 0 = Origem Microfone
                    }

                    recording_buffer.clear();
                    last_partial_len = 0;
                    is_recording = false;

                    // Reseta os estados preditivos para a próxima frase não sofrer vazamento
                    vad.reset_states();
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
            let stream = loop {
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

            let (mut ws_sender, mut ws_receiver) = stream.split();

            // Task para RECEBER dados do Python
            let rx_task = tokio::spawn(async move {
                while let Some(msg_result) = ws_receiver.next().await {
                    match msg_result {
                        Ok(Message::Text(txt)) => {
                            // Rust só imprime o JSON cru agora porque o launcher GUI parseia.
                            println!("\n\x1b[35m[AÇÃO PYTHON]\x1b[0m {}", txt);
                        }
                        Ok(Message::Close(_)) => break,
                        _ => {}
                    }
                }
            });

            while let Some((source_id, is_final, audio_data)) = rx_whisper.recv().await {
                // Header binário: [0x01 = ASR Audio] [SourceID u8] [IsFinal u8] [Float32 PCM...]
                let mut byte_data: Vec<u8> = vec![1, source_id, if is_final { 1 } else { 0 }];
                byte_data.extend(audio_data.iter().flat_map(|&f| f.to_le_bytes()));

                if let Err(e) = ws_sender.send(Message::Binary(byte_data.into())).await {
                    eprintln!(
                        "[WHISPER] Erro ao enviar áudio: {}. Tentando reconectar...",
                        e
                    );
                    break;
                }
            }

            rx_task.abort();
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            continue 'whisper_bridge;
        }
    });

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
                println!(
                    "[DEBUG] Capturados {} blocos do microfone",
                    _debug_frames_captured
                );
            }

            for frame in data.chunks(in_channels) {
                let mono_s = frame[0]; // Pega apenas um canal (mono)
                df_buffer.push(mono_s);

                if df_buffer.len() >= df_size {
                    // OTIMIZAÇÃO DE CPU: Ignorar o modelo de ruído se o quadro for de silêncio absoluto.
                    // Isso poupa inferências ONNX custosas (100x por segundo) quando não há som.
                    let max_amp = df_buffer.iter().copied().fold(0.0f32, f32::max);
                    let min_amp = df_buffer.iter().copied().fold(0.0f32, f32::min);
                    let is_silent = (max_amp - min_amp) < 0.005;

                    if !is_silent {
                        denoiser.process_frame(&mut df_buffer);
                    }

                    for &clean_s in &df_buffer {
                        // 2. Coleta para a IA (Resampling simplificado)
                        current_pos += 1.0;
                        if current_pos >= sample_step {
                            chunk_collector.push(clean_s);
                            current_pos -= sample_step; // Subtrai em vez de zerar

                            if chunk_collector.len() >= 512 {
                                let _ = tx_mic.send(chunk_collector.clone());
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

    let mut sys_chunk_collector = Vec::with_capacity(512);
    let sys_sample_step = out_config.sample_rate.0 as f32 / 16000.0;
    let mut sys_pos = 0.0f32;

    let output_stream = out.build_input_stream(
        &out_config,
        move |data: &[f32], _| {
            for frame in data.chunks(out_channels) {
                let s = frame[0]; // Captura mono do Desktop
                sys_pos += 1.0;
                if sys_pos >= sys_sample_step {
                    sys_chunk_collector.push(s);
                    sys_pos -= sys_sample_step;

                    if sys_chunk_collector.len() >= 512 {
                        let _ = tx_sys.send(sys_chunk_collector.clone());
                        sys_chunk_collector.clear();
                    }
                }
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

    // Mantém os streams vivos enquanto o processo estiver rodando.
    // Evita depender de stdin, que pode estar fechado quando o app é
    // iniciado por launchers GUI (causando encerramento imediato).
    loop {
        thread::sleep(Duration::from_secs(1));
    }

    #[allow(unreachable_code)]
    Ok(())
}
