use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use serde::{Serialize};

#[derive(Debug, Clone, Default)]
pub struct AudioConfig {
    pub mic_name: Option<String>,
    pub meeting_name: Option<String>,
    pub output_name: Option<String>,
}

pub async fn start_bridge(cfg: AudioConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("[IA] Motores ativos. Inicializando streams...");
    let host = cpal::default_host();
    
    // --- SELEÇÃO DE MICROFONE (EU) ---
    let mut mic = if let Some(ref name) = cfg.mic_name {
        if name == "Padrão" {
            host.default_input_device().expect("Mic padrão não encontrado")
        } else {
            host.input_devices()?
                .find(|d| d.name().unwrap_or_default() == *name)
                .ok_or_else(|| format!("Dispositivo de Mic '{}' não encontrado", name))?
        }
    } else {
        host.default_input_device().expect("Mic padrão não encontrado")
    };

    let mic_name_lower = mic.name().unwrap_or_default().to_lowercase();
    
    // Proteção automática se não for especificado: Evita capturar do Virtual Cable se o usuário esqueceu
    if cfg.mic_name.is_none() && (mic_name_lower.contains("cable") || mic_name_lower.contains("virtual")) {
        if let Ok(mut devices) = host.input_devices() {
            if let Some(real_mic) = devices.find(|d| {
                let name = d.name().unwrap_or_default().to_lowercase();
                !name.contains("cable") && !name.contains("virtual") && !name.contains("mix")
            }) {
                mic = real_mic;
            }
        }
    }
    println!("🎙️ Microfone Ativo: {}", mic.name().unwrap_or_default());

    // --- SELEÇÃO DE SAÍDA (TRADUÇÃO) ---
    let out = if let Some(ref name) = cfg.output_name {
        if name == "Padrão" {
            host.default_output_device().expect("Saída padrão zero.")
        } else {
            host.output_devices()?
                .find(|d| d.name().unwrap_or_default() == *name)
                .ok_or_else(|| format!("Dispositivo de Saída '{}' não encontrado", name))?
        }
    } else {
        // Fallback inteligente para "CABLE Input" se disponível
        host.output_devices()?
            .find(|d| d.name().unwrap_or_default().to_lowercase().contains("cable input"))
            .unwrap_or_else(|| {
                println!("[!] Aviso: 'CABLE Input' manual não definido e não auto-detectado. Usando padrão.");
                host.default_output_device().expect("Saída padrão zero.")
            })
    };
    println!("🔊 Saída de Tradução Ativa: {}", out.name().unwrap_or_default());

    // --- SELEÇÃO DE FONTE DE REUNIÃO (LOOPBACK) ---
    let loopback_dev = if let Some(ref name) = cfg.meeting_name {
        if name == "Padrão" {
            host.input_devices()?
                .find(|d| {
                    let n = d.name().unwrap_or_default().to_lowercase();
                    n.contains("cable output") || n.contains("stereo mix") || n.contains("loopback")
                })
        } else {
            host.input_devices()?
                .find(|d| d.name().unwrap_or_default() == *name)
        }
    } else {
        // Fallback inteligente para "CABLE Output" ou Stereo Mix
        host.input_devices()?
            .find(|d| {
                let name = d.name().unwrap_or_default().to_lowercase();
                name.contains("cable output") || name.contains("stereo mix") || name.contains("loopback")
            })
    };

    if let Some(ref dev) = loopback_dev {
        println!("👥 Fonte de Reunião Ativa: {}", dev.name().unwrap_or_default());
    } else {
        println!("⚠️ Aviso: Nenhuma fonte de reunião (loopback) ativa. Você só traduzirá o seu microfone.");
    }

    let config: cpal::StreamConfig = mic.default_input_config()?.into();
    let native_sr = config.sample_rate.0;
    let in_channels = config.channels as usize;

    let out_config: cpal::StreamConfig = out.default_output_config()?.into();
    let out_channels = out_config.channels as usize;

    let (tx, rx) = mpsc::channel::<(usize, Vec<f32>)>();
    let (tx_whisper, mut rx_whisper) = tokio::sync::mpsc::unbounded_channel::<(usize, bool, Vec<f32>)>();
    let (tx_tts, rx_tts) = mpsc::channel::<(u32, Vec<f32>)>();

    struct VadState {
        recording_buffer: Vec<f32>,
        silence_frames: usize,
        is_recording: bool,
        partial_timer: usize,
    }

    impl Default for VadState {
        fn default() -> Self {
            Self {
                recording_buffer: Vec::new(),
                silence_frames: 0,
                is_recording: false,
                partial_timer: 0,
            }
        }
    }

    let last_mic_now = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let last_mic_vad = last_mic_now.clone();
    let last_tts_now = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let last_tts_vad = last_tts_now.clone();
    
    let ptt_active = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ptt_vad = ptt_active.clone();
    let ptt_ws = ptt_active.clone();
    
    // Thread de VAD (Mantemos thread::spawn para prioridade de CPU no áudio)
    thread::spawn(move || {
        let mut states = vec![VadState::default(), VadState::default()];
        let silence_threshold = 0.0050; // Aumentado levemente para evitar ruído de fundo
        let silence_max_frames: usize = 65; 

        while let Ok((source_id, samples)) = rx.recv() {
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
            let last_tts = last_tts_vad.load(std::sync::atomic::Ordering::Relaxed);
            let last_mic = last_mic_vad.load(std::sync::atomic::Ordering::Relaxed);
            let ptt = ptt_vad.load(std::sync::atomic::Ordering::Relaxed);
            
            // --- LÓGICA CONDUCTOR (EXCLUSIVIDADE TOTAL) ---
            if source_id == 0 {
                // Seu microfone SÓ funciona se o Espaço estiver apertado
                if !ptt {
                    let state = &mut states[source_id];
                    if state.is_recording {
                        if state.recording_buffer.len() > 4000 { 
                            let _ = tx_whisper.send((source_id, true, state.recording_buffer.clone()));
                        }
                        state.recording_buffer.clear();
                        state.is_recording = false;
                    }
                    continue; 
                }
            } else if source_id == 1 {
                // A reunião SÓ funciona se você NÃO estiver falando e o Espaço NÃO estiver apertado
                let user_active = now < last_mic + 800 || ptt; 
                let tts_active = now < last_tts + 2500; 
                if user_active || tts_active {
                    continue; 
                }
            }
            
            let state = &mut states[source_id];
            let mut sum_squares = 0.0;
            for &s in &samples { sum_squares += s * s; }
            let rms = (sum_squares / samples.len() as f32).sqrt();
            let mut is_speaking = rms > silence_threshold;

            if source_id == 0 {
                is_speaking = ptt; // No PTT estrito, o botão manda
            }

            if is_speaking {
                state.silence_frames = 0;
                if source_id == 0 {
                    last_mic_vad.store(now, std::sync::atomic::Ordering::Relaxed);
                }
                
                if !state.is_recording {
                    state.is_recording = true;
                    state.recording_buffer.clear();
                    state.partial_timer = 0;
                }
            } else {
                state.silence_frames += 1;
            }

            if state.is_recording {
                state.recording_buffer.extend(&samples);
                state.partial_timer += 1;
                if state.partial_timer >= 25 {
                    if state.recording_buffer.len() > 3200 {
                        let _ = tx_whisper.send((source_id, false, state.recording_buffer.clone()));
                    }
                    state.partial_timer = 0;
                }
                if state.recording_buffer.len() >= 16000 * 15 {
                    let _ = tx_whisper.send((source_id, true, state.recording_buffer.clone()));
                    state.recording_buffer.clear();
                    state.is_recording = false;
                } else if state.silence_frames > silence_max_frames {
                    if state.recording_buffer.len() > 8000 { 
                        let _ = tx_whisper.send((source_id, true, state.recording_buffer.clone()));
                    }
                    state.recording_buffer.clear();
                    state.is_recording = false;
                }
            }
        }
    });

    // Bridge WebSocket (Whisper Python)
    tokio::spawn(async move {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
        loop {
            println!("[DEBUG] Tentando conectar ao servidor Python...");
            let stream = loop {
                match connect_async("ws://127.0.0.1:8000").await {
                    Ok((stream, _)) => {
                        break stream;
                    },
                    Err(_) => tokio::time::sleep(tokio::time::Duration::from_secs(2)).await,
                }
            };
            let (mut ws_sender, mut ws_receiver) = stream.split();
            let tx_tts_clone = tx_tts.clone();
            let ptt_ws_loop = ptt_ws.clone();
            let rx_task = tokio::spawn(async move {
                while let Some(Ok(msg)) = ws_receiver.next().await {
                    if let Message::Binary(bin) = msg {
                        if bin.len() >= 5 && bin[0] == 2 {
                            let sr = u32::from_le_bytes([bin[1], bin[2], bin[3], bin[4]]);
                            let mut tts_audio = Vec::with_capacity(bin[5..].len() / 4);
                            for chunk in bin[5..].chunks_exact(4) {
                                tts_audio.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                            }
                            let _ = tx_tts_clone.send((sr, tts_audio));
                        }
                    } else if let Message::Text(txt) = msg {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&txt) {
                            if json["command"] == "set_ptt" {
                                let val = json["value"].as_bool().unwrap_or(false);
                                ptt_ws_loop.store(val, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                }
            });
            while let Some((sid, is_f, data)) = rx_whisper.recv().await {
                let mut p = vec![sid as u8, if is_f {1} else {0}];
                p.extend(data.iter().flat_map(|&f| f.to_le_bytes()));
                if ws_sender.send(Message::Binary(p.into())).await.is_err() { break; }
            }
            rx_task.abort();
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });

    let rb = HeapRb::<f32>::new(native_sr as usize * 2);
    let (mut prod, mut _cons) = rb.split();
    let tts_rb = HeapRb::<f32>::new(out_config.sample_rate.0 as usize * 15);
    let (mut tts_prod, mut tts_cons) = tts_rb.split();
    let out_sr = out_config.sample_rate.0 as f32;

    let last_tts_prod = last_tts_now.clone();
    thread::spawn(move || {
        while let Ok((sr, tts_audio)) = rx_tts.recv() {
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
            last_tts_prod.store(now, std::sync::atomic::Ordering::Relaxed);
            
            let step = sr as f32 / out_sr;
            let mut pos = 0.0f32;
            while (pos as usize) < tts_audio.len() {
                while tts_prod.is_full() { thread::sleep(Duration::from_millis(5)); }
                let _ = tts_prod.push(tts_audio[pos as usize]);
                pos += step;
                // Atualiza o timestamp continuamente enquanto o áudio entra no pipe
                let now_loop = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
                last_tts_prod.store(now_loop, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });

    let mut chunk_mic = Vec::with_capacity(512);
    let mut chunk_lb = Vec::with_capacity(512);

    let tx_mic = tx.clone();
    let input_stream = mic.build_input_stream(
        &config,
        move |data: &[f32], _| {
            let mut collector = 0.0;
            let mut count = 0;
            for frame in data.chunks(in_channels) {
                let s = frame[0];
                let _ = prod.push(s);
                collector += s;
                count += 1;
                if count >= 3 {
                    chunk_mic.push(collector / 3.0);
                    collector = 0.0;
                    count = 0;
                    if chunk_mic.len() >= 512 {
                        let _ = tx_mic.send((0, chunk_mic.clone()));
                        chunk_mic.clear();
                    }
                }
            }
        },
        |e| eprintln!("Erro Mic: {}", e),
        None,
    )?;

    let last_tts_lb = last_tts_now.clone();
    let last_tts_out = last_tts_now.clone();

    let _stream_lb = if let Some(dev) = loopback_dev {
        let lb_config: cpal::StreamConfig = dev.default_input_config()?.into();
        let lb_channels = lb_config.channels as usize;
        let tx_lb = tx.clone();
        Some(dev.build_input_stream(
            &lb_config,
            move |data: &[f32], _| {
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
                let last = last_tts_lb.load(std::sync::atomic::Ordering::Relaxed);
                let mute = now < last + 2500; // Mudo por 2.5s para garantir limpeza total do cabo virtual
                
                let mut col = 0.0;
                let mut cnt = 0;
                for f in data.chunks(lb_channels) {
                    let s = if mute { 0.0 } else { f[0] };
                    col += s;
                    cnt += 1;
                    if cnt >= 3 {
                        chunk_lb.push(col / 3.0);
                        col = 0.0;
                        cnt = 0;
                        if chunk_lb.len() >= 512 {
                            let _ = tx_lb.send((1, chunk_lb.clone()));
                            chunk_lb.clear();
                        }
                    }
                }
            },
            |e| eprintln!("Erro Loopback: {}", e),
            None,
        )?)
    } else { None };

    let output_stream = out.build_output_stream(
        &out_config,
        move |data: &mut [f32], _| {
            let mut playing = false;
            for frame in data.chunks_mut(out_channels) {
                let s = tts_cons.pop().unwrap_or(0.0);
                if s.abs() > 0.0001 { playing = true; }
                for sample in frame { *sample = s; }
            }
            if playing {
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
                last_tts_out.store(now, std::sync::atomic::Ordering::Relaxed);
            }
        },
        |e| eprintln!("Erro Output: {}", e),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;
    if let Some(ref s) = _stream_lb { s.play()?; }

    println!("[IA] Motores ativos. Arquitetura Dual VAD (Eu + Reunião)...");
    loop { tokio::time::sleep(Duration::from_millis(1000)).await; }
}
