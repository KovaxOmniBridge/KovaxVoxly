mod driver;
mod audio;
pub mod engine;

const FLEXASIO_BINARY: &[u8] = include_bytes!("../FlexASIO-1.10b.exe");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // No ort rc.12, o ambiente é gerenciado automaticamente ao criar a primeira Session.
    
    // 1. Configura o ambiente de driver (FlexASIO)
    driver::setup_environment(FLEXASIO_BINARY)?;

    // 2. Inicia o motor de áudio e as threads de IA
    audio::start_bridge()?;

    Ok(())
}
