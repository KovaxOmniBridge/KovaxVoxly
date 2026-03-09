mod audio;
mod driver;
pub mod engine;

const FLEXASIO_BINARY: &[u8] = include_bytes!("../tools/FlexASIO-1.10b.exe");
const VBCABLE_ZIP: &[u8] = include_bytes!("../tools/vbcable.zip");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // No ort rc.12, o ambiente é gerenciado automaticamente ao criar a primeira Session.

    // 1. Configura o ambiente de driver (FlexASIO e Virtual Cable)
    driver::setup_environment(FLEXASIO_BINARY, VBCABLE_ZIP)?;

    // 2. Inicia o motor de áudio e as threads de IA
    audio::start_bridge()?;

    Ok(())
}
