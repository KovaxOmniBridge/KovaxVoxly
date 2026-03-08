use ringbuf::{Consumer, HeapRb, Producer};
use std::collections::VecDeque;

/// Um buffer circular infinito para fluxo de áudio limpo entre a captura e a saída.
pub struct AudioStreamBuffer {
    pub prod:
        Producer<f32, std::sync::Arc<ringbuf::SharedRb<f32, Vec<std::mem::MaybeUninit<f32>>>>>,
    pub cons:
        Consumer<f32, std::sync::Arc<ringbuf::SharedRb<f32, Vec<std::mem::MaybeUninit<f32>>>>>,
}

impl AudioStreamBuffer {
    pub fn new(capacity: usize) -> Self {
        let rb = HeapRb::<f32>::new(capacity);
        let (prod, cons) = rb.split();
        Self { prod, cons }
    }
}

/// Um buffer de "Pré-gravação" (Pre-roll).
/// Ele mantém os últimos N samples continuamente girando.
/// Quando o VAD detecta voz, nós já temos guardado o início da respiração da pessoa
/// para evitar que a primeira sílaba seja cortada.
pub struct PreRollBuffer {
    queue: VecDeque<f32>,
    max_capacity: usize,
}

impl PreRollBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity),
            max_capacity: capacity,
        }
    }

    pub fn push_samples(&mut self, samples: &[f32]) {
        for &s in samples {
            if self.queue.len() >= self.max_capacity {
                self.queue.pop_front();
            }
            self.queue.push_back(s);
        }
    }

    pub fn drain_to(&mut self, output: &mut Vec<f32>) {
        output.extend(self.queue.iter());
        self.queue.clear();
    }
}
