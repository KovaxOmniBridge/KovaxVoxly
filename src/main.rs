use clap::Parser;
use serde::Serialize;
mod audio;
mod driver;

const FLEXASIO_BINARY: &[u8] = include_bytes!("../tools/FlexASIO-1.10b.exe");
const VBCABLE_ZIP: &[u8] = include_bytes!("../tools/vbcable.zip");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// List all available audio devices and exit
    #[arg(long)]
    list_devices: bool,

    /// Name of the physical microphone to use (Input)
    #[arg(long)]
    mic: Option<String>,

    /// Name of the meeting audio source (Loopback/CABLE Output)
    #[arg(long)]
    meeting: Option<String>,

    /// Name of the translation destination (CABLE Input)
    #[arg(long)]
    output: Option<String>,
}

#[derive(Serialize)]
struct DeviceList {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.list_devices {
        use cpal::traits::{DeviceTrait, HostTrait};
        let host = cpal::default_host();
        
        let inputs: Vec<String> = match host.input_devices() {
            Ok(iter) => iter.filter_map(|d| d.name().ok()).collect(),
            Err(_) => Vec::new(),
        };
            
        let outputs: Vec<String> = match host.output_devices() {
            Ok(iter) => iter.filter_map(|d| d.name().ok()).collect(),
            Err(_) => Vec::new(),
        };
        
        let list = DeviceList { inputs, outputs };
        if let Ok(json) = serde_json::to_string(&list) {
            println!("{}", json);
        }
        return Ok(());
    }

    // 2. Configura o ambiente de driver (FlexASIO e Virtual Cable)
    driver::setup_environment(FLEXASIO_BINARY, VBCABLE_ZIP)?;

    // 3. Inicia o motor de áudio com as seleções manuais (se houver)
    let config = audio::AudioConfig {
        mic_name: args.mic,
        meeting_name: args.meeting,
        output_name: args.output,
    };

    audio::start_bridge(config).await?;

    Ok(())
}
