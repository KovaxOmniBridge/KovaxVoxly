use crate::engine; 
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use std::io;
use std::sync::mpsc;
use std::thread;
use std::collections::VecDeque;

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
    let (tx_whisper, mut rx_whisper) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();

    // --- THREAD DE INTELIGÊNCIA ---
    thread::spawn(move || {
        let mut vad = engine::Vad::new("silero_vad.onnx").expect("Erro VAD");
        let mut diarizer = engine::Diarizer::new("segmentation-3.0.onnx").expect("Erro Diarizer");

        let mut pre_roll = VecDeque::with_capacity(8000); 
        let mut recording_buffer = Vec::new();
        let mut is_recording = false;
        let mut silence_frames = 0;

        println!("[IA] Motores ativos. Fale no microfone...");

        while let Ok(samples) = rx.recv() {
            for &s in &samples {
                if pre_roll.len() >= 8000 { pre_roll.pop_front(); }
                pre_roll.push_back(s);
            }

            let probability = vad.is_speech(samples.clone());

            // Ajuste de Sensibilidade: 0.30 é ideal para microfones comuns
            if probability > 0.30 { 
                if !is_recording {
                    is_recording = true;
                    recording_buffer.extend(pre_roll.iter());
                }
                recording_buffer.extend(samples);
                silence_frames = 0;

                let speaker_id = diarizer.identify(&recording_buffer);
                print!("\r[!] VOZ DETECTADA: Locutor {} | Prob: {:.0}% | Gravando: {:.1}s    ", 
                    speaker_id, probability * 100.0, recording_buffer.len() as f32 / 16000.0);
                use std::io::Write; let _ = std::io::stdout().flush();
            } else if is_recording {
                recording_buffer.extend(samples);
                silence_frames += 1;

                if silence_frames > 30 { 
                    println!("\n[✓] Frase finalizada. Enviando para Whisper...");
                    let _ = tx_whisper.send(recording_buffer.clone());
                    recording_buffer.clear();
                    is_recording = false;
                }
            }
        }
    });

    tokio::spawn(async move {
        while let Some(audio_data) = rx_whisper.recv().await {
            println!("[WHISPER] Processando {} bytes de áudio...", audio_data.len() * 4);
        }
    });

    let rb = HeapRb::<f32>::new(native_sr as usize * 2);
    let (mut prod, mut cons) = rb.split();
    
    let mut chunk_collector = Vec::with_capacity(512);
    let sample_step = native_sr as f32 / 16000.0;
    let mut current_pos = 0.0f32;

    let input_stream = mic.build_input_stream(
        &config,
        move |data: &[f32], _| {
            for &s in data {
                df_buffer.push(s);

                if df_buffer.len() >= df_size {
                    // 1. Limpa o áudio (O que você ouve no fone)
                    denoiser.process_frame(&mut df_buffer);

                    for &clean_s in &df_buffer {
                        let _ = prod.push(clean_s); 

                        // 2. Coleta para a IA (Resampling simplificado)
                        // Se native_sr é 48000, pegamos 1 a cada 3 samples para dar 16000
                        current_pos += 1.0;
                        if current_pos >= sample_step {
                            chunk_collector.push(clean_s);
                            current_pos = 0.0; // Reset simples

                            if chunk_collector.len() >= 512 {
                                // FORÇAR ENVIO: O Rio agora deságua na IA
                                let _ = tx.send(chunk_collector.clone());
                                chunk_collector.clear();
                            }
                        }
                    }
                    df_buffer.clear();
                }
            }
        },

        |e| eprintln!("Erro Mic: {}", e),
        None
    )?;


    let output_stream = out.build_output_stream(
        &config,
        move |data: &mut [f32], _| {
            for s in data.iter_mut() { *s = cons.pop().unwrap_or(0.0); }
        },

        |e| eprintln!("Erro Saída: {}", e),
        None
    )?;

    input_stream.play()?;
    output_stream.play()?;

    println!("\n[VIVO] Voxly: {}Hz nativos.", native_sr);
    io::stdin().read_line(&mut String::new())?;
    Ok(())
}
