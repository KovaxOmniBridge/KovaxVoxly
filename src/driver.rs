use cpal::traits::{DeviceTrait, HostTrait};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;

pub fn setup_environment(
    flexasio_binary: &[u8],
    vbcable_binary: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("[*] Verificando ambiente de driver...");

    if !is_flexasio_installed() {
        install_flexasio_smart(flexasio_binary)?;
    } else {
        println!("[+] Driver FlexASIO já instalado.");
    }

    if !is_vbcable_installed() {
        install_vbcable_smart(vbcable_binary)?;
    } else {
        println!("[+] Driver VB-Cable já instalado.");
    }

    let host = cpal::default_host();
    let mic_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_else(|| "Default".to_string());
    let out_name = host
        .default_output_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_else(|| "Default".to_string());

    let user_profile = std::env::var("USERPROFILE")?;
    let config_path = Path::new(&user_profile).join("FlexASIO.toml");

    // MELHORIA: Adicionado aspas duplas escapadas para nomes de dispositivos e latência otimizada
    let config_content = format!(
        r#"backend = "Windows WASAPI"
[input]
device = "{}"
[output]
device = "{}"
# Otimização para baixa latência necessária para o DeepFilterNet
wasapiExclusiveMode = true
wasapiAutoConvert = true
"#,
        mic_name, out_name
    );

    fs::write(config_path, config_content)?;
    Ok(())
}
fn is_flexasio_installed() -> bool {
    // Verifica caminhos comuns e o executável de configuração
    let common_paths = [
        "C:\\Program Files\\FlexASIO\\FlexASIO.dll",
        "C:\\Program Files (x86)\\FlexASIO\\FlexASIO.dll",
        "C:\\Program Files\\FlexASIO\\x64\\FlexASIO.dll",
        "C:\\Program Files\\FlexASIO\\FlexASIOGUI.exe",
    ];

    common_paths.iter().any(|p| Path::new(p).exists())
}

fn install_flexasio_smart(binary: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let temp_installer = std::env::temp_dir().join("voxly_installer_v1.exe");

    // EVITA ERRO 32: Só cria o arquivo se ele não existir fisicamente no Temp
    if !temp_installer.exists() {
        match fs::File::create(&temp_installer) {
            Ok(mut file) => {
                let _ = file.write_all(binary);
            }
            Err(_) => { /* Se não puder criar (está em uso), ignoramos e tentamos rodar o que já está lá */
            }
        }
    }

    println!("[!] Driver não detectado. Instalando FlexASIO (solicitando admin)...");

    // Executa com privilégios de Administrador
    let _ = Command::new("powershell")
        .args(&[
            "-NoProfile",
            "-Command",
            &format!(
                "Start-Process -FilePath '{}' -ArgumentList '/VERYSILENT' -Verb RunAs -Wait",
                temp_installer.display()
            ),
        ])
        .status();

    // Não tentamos apagar imediatamente para evitar conflito com o processo de instalação
    Ok(())
}

fn is_vbcable_installed() -> bool {
    let common_paths = [
        "C:\\Program Files\\VB\\CABLE\\VBCABLE_ControlPanel.exe",
        "C:\\Program Files (x86)\\VB\\CABLE\\VBCABLE_ControlPanel.exe",
    ];
    common_paths.iter().any(|p| Path::new(p).exists())
}

fn install_vbcable_smart(binary_zip: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let temp_zip = std::env::temp_dir().join("vbcable_voxly.zip");
    let extract_dir = std::env::temp_dir().join("vbcable_voxly_extracted");

    println!("[!] Virtual Cable não detectado. Extraindo e instalando (solicitando admin)...");

    if !temp_zip.exists() {
        match fs::File::create(&temp_zip) {
            Ok(mut file) => {
                let _ = file.write_all(binary_zip);
            }
            Err(_) => {}
        }
    }

    // Usando PowerShell para extrair nativamente
    let _ = Command::new("powershell")
        .args(&[
            "-NoProfile",
            "-Command",
            &format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                temp_zip.display(),
                extract_dir.display()
            ),
        ])
        .status();

    let setup_exe = extract_dir.join("VBCABLE_Setup_x64.exe");
    if setup_exe.exists() {
        // -i para install, -h para hidden, rodando como Administrador
        let _ = Command::new("powershell")
            .args(&[
                "-NoProfile",
                "-Command",
                &format!(
                    "Start-Process -FilePath '{}' -ArgumentList '-i -h' -Verb RunAs -Wait",
                    setup_exe.display()
                ),
            ])
            .status();
    }

    Ok(())
}
