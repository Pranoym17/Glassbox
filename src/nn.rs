use crate::Tensor;
use crate::autograd::{Tape, TapeError, TensorId};

pub trait Module {
    type Input;

    fn parameters(&self) -> Vec<TensorId>;
    fn forward(&self, tape: &mut Tape, input: Self::Input) -> Result<TensorId, TapeError>;
}

pub struct Initializer {
    state: u64,
    spare: Option<f32>,
}

impl Initializer {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
            spare: None,
        }
    }

    fn uniform(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        ((x >> 40) as f32 + 1.0) / 16_777_217.0
    }

    pub fn normal(&mut self, standard_deviation: f32) -> f32 {
        if let Some(value) = self.spare.take() {
            return value * standard_deviation;
        }
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos() * standard_deviation
    }
}

pub struct Linear {
    weight: TensorId,
    bias: TensorId,
}

impl Linear {
    pub fn new(
        tape: &mut Tape,
        input_features: usize,
        output_features: usize,
        initializer: &mut Initializer,
    ) -> Result<Self, TapeError> {
        let weight = Tensor::from_vec(
            vec![input_features, output_features],
            (0..input_features * output_features)
                .map(|_| initializer.normal(0.02))
                .collect(),
        )?;
        Ok(Self {
            weight: tape.leaf(weight),
            bias: tape.leaf(Tensor::zeros(vec![output_features])?),
        })
    }
}

impl Module for Linear {
    type Input = TensorId;

    fn parameters(&self) -> Vec<TensorId> {
        vec![self.weight, self.bias]
    }

    fn forward(&self, tape: &mut Tape, input: TensorId) -> Result<TensorId, TapeError> {
        let projected = tape.matmul(input, self.weight)?;
        tape.add(projected, self.bias)
    }
}

pub struct LayerNorm {
    gamma: TensorId,
    beta: TensorId,
    epsilon: f32,
}

impl LayerNorm {
    pub fn new(tape: &mut Tape, features: usize, epsilon: f32) -> Result<Self, TapeError> {
        Ok(Self {
            gamma: tape.leaf(Tensor::ones(vec![features])?),
            beta: tape.leaf(Tensor::zeros(vec![features])?),
            epsilon,
        })
    }
}

impl Module for LayerNorm {
    type Input = TensorId;

    fn parameters(&self) -> Vec<TensorId> {
        vec![self.gamma, self.beta]
    }

    fn forward(&self, tape: &mut Tape, input: TensorId) -> Result<TensorId, TapeError> {
        tape.layer_norm(input, self.gamma, self.beta, self.epsilon)
    }
}

pub struct Embedding {
    table: TensorId,
}

impl Embedding {
    pub fn new(
        tape: &mut Tape,
        entries: usize,
        features: usize,
        initializer: &mut Initializer,
    ) -> Result<Self, TapeError> {
        let table = Tensor::from_vec(
            vec![entries, features],
            (0..entries * features)
                .map(|_| initializer.normal(0.02))
                .collect(),
        )?;
        Ok(Self {
            table: tape.leaf(table),
        })
    }
}

impl Module for Embedding {
    type Input = Vec<usize>;

    fn parameters(&self) -> Vec<TensorId> {
        vec![self.table]
    }

    fn forward(&self, tape: &mut Tape, indices: Vec<usize>) -> Result<TensorId, TapeError> {
        tape.embedding(self.table, indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_is_deterministic() {
        let mut left_tape = Tape::new();
        let mut right_tape = Tape::new();
        let mut left_rng = Initializer::new(42);
        let mut right_rng = Initializer::new(42);
        let left = Linear::new(&mut left_tape, 3, 4, &mut left_rng).unwrap();
        let right = Linear::new(&mut right_tape, 3, 4, &mut right_rng).unwrap();
        for (left_id, right_id) in left.parameters().iter().zip(right.parameters()) {
            assert_eq!(
                left_tape.value(*left_id).unwrap(),
                right_tape.value(right_id).unwrap()
            );
        }
    }

    #[test]
    fn modules_forward_with_expected_shapes() {
        let mut tape = Tape::new();
        let mut initializer = Initializer::new(7);
        let embedding = Embedding::new(&mut tape, 5, 3, &mut initializer).unwrap();
        let norm = LayerNorm::new(&mut tape, 3, 1e-5).unwrap();
        let linear = Linear::new(&mut tape, 3, 2, &mut initializer).unwrap();
        let selected = embedding.forward(&mut tape, vec![1, 4, 1]).unwrap();
        let normalized = norm.forward(&mut tape, selected).unwrap();
        let output = linear.forward(&mut tape, normalized).unwrap();

        assert_eq!(tape.value(selected).unwrap().shape(), &[3, 3]);
        assert_eq!(tape.value(normalized).unwrap().shape(), &[3, 3]);
        assert_eq!(tape.value(output).unwrap().shape(), &[3, 2]);
        assert_eq!(embedding.parameters().len(), 1);
        assert_eq!(norm.parameters().len(), 2);
        assert_eq!(linear.parameters().len(), 2);
    }
}
