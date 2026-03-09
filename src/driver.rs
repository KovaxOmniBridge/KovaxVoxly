use cpal::traits::{DeviceTrait, HostTrait};

pub fn setup_environment() -> Result<(), Box<dyn std::error::Error>> {
    println!("[*] Inicializando motor WASAPI puro...");
    let host = cpal::default_host();
    let mic_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_else(|| "Default".to_string());
    let out_name = host
        .default_output_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_else(|| "Default".to_string());

    println!("[+] Microfone Padrão: {}", mic_name);
    println!("[+] Saída Padrão (Loopback Tracker): {}", out_name);

    Ok(())
}
