use ort::session::Session;
use ort::value::Value;

pub struct Denoiser {
    enc_session: Session,
    erb_session: Session,
    df_session: Session,
    states: Vec<f32>, 
}

impl Denoiser {
    pub fn new(_sample_rate: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let enc_session = Session::builder()?.commit_from_file("enc.onnx")?;
        let erb_session = Session::builder()?.commit_from_file("erb_dec.onnx")?;
        let df_session = Session::builder()?.commit_from_file("df_dec.onnx")?;

        Ok(Self {
            enc_session,
            erb_session,
            df_session,
            states: vec![0.0; 256], 
        })
    }

    pub fn process_frame(&mut self, buffer: &mut [f32]) {
        let input_shape = [1usize, 1usize, buffer.len()];
        let input_tensor = Value::from_array((input_shape, buffer.to_vec())).unwrap();

        // 1. PASSO: ENCODER - Removido .unwrap() do final do macro
        let enc_inputs = ort::inputs![
            "input" => input_tensor.view()
        ];

        if let Ok(enc_outs) = self.enc_session.run(enc_inputs) {
            if let Some(latent) = enc_outs.get("latent") {

                // 2. PASSO: ERB DECODER - Removido .unwrap() do final do macro
                let erb_inputs = ort::inputs![
                    "latent" => latent,
                    "states" => Value::from_array(([1usize, 1usize, 256usize], self.states.clone())).unwrap()
                ];
                
                if let Ok(erb_outs) = self.erb_session.run(erb_inputs) {
                    if let Some(erb_val) = erb_outs.get("erb") {

                        // 3. PASSO: DF DECODER - Removido .unwrap() do final do macro
                        let df_inputs = ort::inputs![
                            "latent" => latent,
                            "erb" => erb_val
                        ];

                        if let Ok(df_outs) = self.df_session.run(df_inputs) {
                            if let Some(out_value) = df_outs.get("output") {
                                let prob_tensor = out_value.try_extract_tensor::<f32>().unwrap();
                                let clean_data = prob_tensor.1;
                                
                                for (i, &clean_sample) in clean_data.iter().enumerate() {
                                    if i < buffer.len() { buffer[i] = clean_sample; }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn frame_size(&self) -> usize {
        480 
    }
}
