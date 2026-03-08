use ort::session::Session;
use ort::value::Value;

pub struct Diarizer {
    session: Session,
    history: Vec<f32>,
}

impl Diarizer {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(model_path)?;

        Ok(Self {
            session,
            history: Vec::new(),
        })
    }

    pub fn identify(&mut self, chunk: &[f32]) -> usize {
        self.history.extend_from_slice(chunk);

        if self.history.len() < 8000 {
            return 0;
        }

        if self.history.len() > 16000 * 5 {
            self.history.drain(0..16000);
        }

        let input_shape = [1, 1, self.history.len()];
        // Criamos o valor explicitamente
        let input_tensor = Value::from_array((input_shape, self.history.clone())).unwrap();
        let inputs = ort::inputs!["audio" => input_tensor];

        let mut speaker_id = 0;

        // Abrimos um escopo ou processamos o resultado imediatamente para liberar o borrow de self
        if let Ok(outputs) = self.session.run(inputs) {
            if let Some(logits) = outputs.get("segmentation") {
                if let Ok(prob_tensor) = logits.try_extract_tensor::<f32>() {
                    // Chamamos a função estática ou passamos apenas os dados necessários
                    speaker_id = Self::extract_dominant_speaker(prob_tensor.1);
                }
            }
        }

        speaker_id
    }

    // Transformada em função associada (estática) para não precisar de &self
    // Isso resolve o erro de "cannot borrow self as immutable"
    fn extract_dominant_speaker(data: &[f32]) -> usize {
        let mut max_val = -1.0;
        let mut speaker_id = 0;

        // O Pyannote costuma retornar probabilidades no tempo para cada locutor
        for (i, &val) in data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                speaker_id = i % 7;
            }
        }
        speaker_id
    }

    pub fn reset_history(&mut self) {
        self.history.clear();
    }
}
