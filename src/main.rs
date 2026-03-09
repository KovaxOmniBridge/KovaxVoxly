mod audio;
mod driver;
pub mod engine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configura log de devices WASAPI
    driver::setup_environment()?;

    // 2. Inicia o motor de áudio dual loopback
    audio::start_bridge()?;

    Ok(())
}
