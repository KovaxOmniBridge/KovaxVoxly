use ort::session::Session;
use ort::value::Value;

pub struct Vad {
    session: Session,
    state_data: Vec<f32>,
}

impl Vad {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(model_path)?;

        Ok(Self {
            session,
            state_data: vec![0.0; 2 * 1 * 128], // Estado único do Silero v4
        })
    }

    pub fn is_speech(&mut self, samples: Vec<f32>) -> f32 {
        // 'sr' no Silero v4 geralmente aceita um tensor 1D vazio ou [1] com o valor.
        let inputs = ort::inputs![
            "input" => Value::from_array(([1usize, samples.len()], samples)).unwrap(),
            "sr" => Value::from_array(([1usize], vec![16000_i64])).unwrap(),
            "state" => Value::from_array(([2usize, 1usize, 128usize], self.state_data.clone())).unwrap()
        ];

        match self.session.run(inputs) {
            Ok(outputs) => {
                if let Some(state_out) = outputs.get("stateN") {
                    let (_, data) = state_out.try_extract_tensor::<f32>().unwrap();
                    self.state_data = data.to_vec();
                }

                if let Some(prob_value) = outputs.get("output") {
                    let prob_tensor = prob_value.try_extract_tensor::<f32>().unwrap();
                    let probability = *prob_tensor.1.first().unwrap_or(&0.0);

                    return probability;
                }
            }
            Err(e) => {
                eprintln!("[DEBUG VAD ERRO] Falha na inferência VAD: {:?}", e);
            }
        }
        0.0
    }

    pub fn reset_states(&mut self) {
        self.state_data.fill(0.0);
    }
}
