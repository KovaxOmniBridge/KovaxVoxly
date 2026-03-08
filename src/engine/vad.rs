use ort::session::Session;
use ort::value::Value;

pub struct Vad {
    session: Session,
    h_data: Vec<f32>,
    c_data: Vec<f32>,
    sr_data: Vec<i64>,
}

impl Vad {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            h_data: vec![0.0; 2 * 1 * 64], // Estado oculto (Hidden state)
            c_data: vec![0.0; 2 * 1 * 64], // Estado de célula (Cell state)
            sr_data: vec![16000],
        })
    }

    pub fn is_speech(&mut self, samples: Vec<f32>) -> f32 {
        let inputs = ort::inputs![
            "input" => Value::from_array(([1usize, 512usize], samples)).unwrap(),
            "sr" => Value::from_array(([1usize], self.sr_data.clone())).unwrap(),
            "h" => Value::from_array(([2usize, 1usize, 64usize], self.h_data.clone())).unwrap(),
            "c" => Value::from_array(([2usize, 1usize, 64usize], self.c_data.clone())).unwrap()
        ];

        // Usamos session.run_mut() ou session.run() se for thread-safe
        if let Ok(outputs) = self.session.run(inputs) {
            
            // 1. ATUALIZAÇÃO DA MEMÓRIA (O "pulo do gato")
            // Pegamos os novos estados 'hn' e 'cn' e salvamos para a próxima iteração
            if let Some(h_out) = outputs.get("hn") {
                let (_, data) = h_out.try_extract_tensor::<f32>().unwrap();
                self.h_data = data.to_vec();
            }
            if let Some(c_out) = outputs.get("cn") {
                let (_, data) = c_out.try_extract_tensor::<f32>().unwrap();
                self.c_data = data.to_vec();
            }

            // 2. RETORNO DA PROBABILIDADE
            if let Some(prob_value) = outputs.get("output") {
                let prob_tensor = prob_value.try_extract_tensor::<f32>().unwrap();
                let probability = *prob_tensor.1.first().unwrap_or(&0.0);
                
                // Debug opcional: descomente para ver os números no terminal
                // println!("Prob: {:.4}", probability); 
                
                return probability;
            }
        }
        0.0
    }
}
