use crate::Tensor;
use crate::autograd::{Tape, TapeError, TensorId};
use crate::transformer::causal_attention_tape;
use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

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

#[derive(Debug)]
pub enum GptError {
    Tape(TapeError),
    UnsupportedHeadCount(usize),
    InvalidConfiguration(&'static str),
    EmptyBatch,
    EmptySequence,
    RaggedBatch,
    TargetBatchMismatch,
    SequenceTooLong { length: usize, block_size: usize },
    Io(std::io::Error),
    InvalidCheckpoint(String),
}

impl fmt::Display for GptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tape(error) => write!(f, "{error}"),
            Self::UnsupportedHeadCount(count) => {
                write!(
                    f,
                    "this GPT implementation supports n_head=1, received {count}"
                )
            }
            Self::InvalidConfiguration(message) => write!(f, "{message}"),
            Self::EmptyBatch => write!(f, "token batch must not be empty"),
            Self::EmptySequence => write!(f, "token sequences must not be empty"),
            Self::RaggedBatch => write!(f, "all token sequences must have the same length"),
            Self::TargetBatchMismatch => {
                write!(
                    f,
                    "targets must match the input batch and sequence dimensions"
                )
            }
            Self::SequenceTooLong { length, block_size } => {
                write!(
                    f,
                    "sequence length {length} exceeds block size {block_size}"
                )
            }
            Self::Io(error) => write!(f, "{error}"),
            Self::InvalidCheckpoint(message) => write!(f, "invalid checkpoint: {message}"),
        }
    }
}

impl std::error::Error for GptError {}

impl From<TapeError> for GptError {
    fn from(error: TapeError) -> Self {
        Self::Tape(error)
    }
}
impl From<std::io::Error> for GptError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

struct TransformerBlock {
    first_norm: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    second_norm: LayerNorm,
    mlp_input: Linear,
    mlp_output: Linear,
}

impl TransformerBlock {
    fn new(
        tape: &mut Tape,
        features: usize,
        initializer: &mut Initializer,
    ) -> Result<Self, TapeError> {
        Ok(Self {
            first_norm: LayerNorm::new(tape, features, 1e-5)?,
            query: Linear::new(tape, features, features, initializer)?,
            key: Linear::new(tape, features, features, initializer)?,
            value: Linear::new(tape, features, features, initializer)?,
            attention_output: Linear::new(tape, features, features, initializer)?,
            second_norm: LayerNorm::new(tape, features, 1e-5)?,
            mlp_input: Linear::new(tape, features, features * 4, initializer)?,
            mlp_output: Linear::new(tape, features * 4, features, initializer)?,
        })
    }

    fn parameters(&self) -> Vec<TensorId> {
        let mut parameters = self.first_norm.parameters();
        parameters.extend(self.query.parameters());
        parameters.extend(self.key.parameters());
        parameters.extend(self.value.parameters());
        parameters.extend(self.attention_output.parameters());
        parameters.extend(self.second_norm.parameters());
        parameters.extend(self.mlp_input.parameters());
        parameters.extend(self.mlp_output.parameters());
        parameters
    }

    fn forward(&self, tape: &mut Tape, input: TensorId) -> Result<TensorId, TapeError> {
        let normalized = self.first_norm.forward(tape, input)?;
        let query = self.query.forward(tape, normalized)?;
        let key = self.key.forward(tape, normalized)?;
        let value = self.value.forward(tape, normalized)?;
        let attention = causal_attention_tape(tape, query, key, value)?;
        let attention = self.attention_output.forward(tape, attention)?;
        let residual = tape.add(input, attention)?;
        let normalized = self.second_norm.forward(tape, residual)?;
        let hidden = self.mlp_input.forward(tape, normalized)?;
        let hidden = tape.gelu(hidden)?;
        let output = self.mlp_output.forward(tape, hidden)?;
        tape.add(residual, output)
    }
}

pub struct Gpt {
    tape: Tape,
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    output: Linear,
    tape_checkpoint: usize,
    vocab_size: usize,
    block_size: usize,
}

impl Gpt {
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
    ) -> Result<Self, GptError> {
        Self::new_with_seed(vocab_size, block_size, n_layer, n_head, n_embd, 1337)
    }

    pub fn new_with_seed(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
        seed: u64,
    ) -> Result<Self, GptError> {
        if n_head != 1 {
            return Err(GptError::UnsupportedHeadCount(n_head));
        }
        if vocab_size == 0 || block_size == 0 || n_embd == 0 {
            return Err(GptError::InvalidConfiguration(
                "vocab_size, block_size, and n_embd must be positive",
            ));
        }
        let mut tape = Tape::new();
        let mut initializer = Initializer::new(seed);
        let token_embedding = Embedding::new(&mut tape, vocab_size, n_embd, &mut initializer)?;
        let position_embedding = Embedding::new(&mut tape, block_size, n_embd, &mut initializer)?;
        let blocks = (0..n_layer)
            .map(|_| TransformerBlock::new(&mut tape, n_embd, &mut initializer))
            .collect::<Result<Vec<_>, _>>()?;
        let final_norm = LayerNorm::new(&mut tape, n_embd, 1e-5)?;
        let output = Linear::new(&mut tape, n_embd, vocab_size, &mut initializer)?;
        let tape_checkpoint = tape.entry_count();
        Ok(Self {
            tape,
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            output,
            tape_checkpoint,
            vocab_size,
            block_size,
        })
    }

    pub fn parameters(&self) -> Vec<TensorId> {
        let mut parameters = self.token_embedding.parameters();
        parameters.extend(self.position_embedding.parameters());
        for block in &self.blocks {
            parameters.extend(block.parameters());
        }
        parameters.extend(self.final_norm.parameters());
        parameters.extend(self.output.parameters());
        parameters
    }
    pub fn parameter_values(&self) -> Result<Vec<Tensor>, GptError> {
        self.parameters()
            .into_iter()
            .map(|id| self.tape.value(id).cloned().map_err(GptError::from))
            .collect()
    }
    pub fn tape_entry_count(&self) -> usize {
        self.tape.entry_count()
    }
    pub fn reset_tape(&mut self) -> Result<(), GptError> {
        Ok(self.tape.reset(self.tape_checkpoint)?)
    }

    fn validate_tokens(&self, tokens: &[Vec<usize>]) -> Result<usize, GptError> {
        let Some(first) = tokens.first() else {
            return Err(GptError::EmptyBatch);
        };
        if first.is_empty() {
            return Err(GptError::EmptySequence);
        }
        let sequence_length = first.len();
        if tokens
            .iter()
            .any(|sequence| sequence.len() != sequence_length)
        {
            return Err(GptError::RaggedBatch);
        }
        if sequence_length > self.block_size {
            return Err(GptError::SequenceTooLong {
                length: sequence_length,
                block_size: self.block_size,
            });
        }
        Ok(sequence_length)
    }

    fn forward_sequence(&mut self, tokens: &[usize]) -> Result<TensorId, GptError> {
        let positions = (0..tokens.len()).collect();
        let token_values = self
            .token_embedding
            .forward(&mut self.tape, tokens.to_vec())?;
        let position_values = self.position_embedding.forward(&mut self.tape, positions)?;
        let mut hidden = self.tape.add(token_values, position_values)?;
        for block in &self.blocks {
            hidden = block.forward(&mut self.tape, hidden)?;
        }
        hidden = self.final_norm.forward(&mut self.tape, hidden)?;
        Ok(self.output.forward(&mut self.tape, hidden)?)
    }

    pub fn forward(&mut self, tokens: &[Vec<usize>]) -> Result<Tensor, GptError> {
        self.reset_tape()?;
        let result = (|| {
            let sequence_length = self.validate_tokens(tokens)?;
            let mut logits = Vec::with_capacity(tokens.len() * sequence_length * self.vocab_size);
            for sequence in tokens {
                let output = self.forward_sequence(sequence)?;
                logits.extend_from_slice(self.tape.value(output)?.data());
            }
            Ok(
                Tensor::from_vec(vec![tokens.len(), sequence_length, self.vocab_size], logits)
                    .map_err(TapeError::from)?,
            )
        })();
        self.reset_tape()?;
        result
    }

    pub fn loss(
        &mut self,
        tokens: &[Vec<usize>],
        targets: &[Vec<usize>],
    ) -> Result<TensorId, GptError> {
        let sequence_length = self.validate_tokens(tokens)?;
        if targets.len() != tokens.len()
            || targets
                .iter()
                .any(|sequence| sequence.len() != sequence_length)
        {
            return Err(GptError::TargetBatchMismatch);
        }
        let mut total = None;
        for (sequence, target) in tokens.iter().zip(targets) {
            let logits = self.forward_sequence(sequence)?;
            let sequence_loss = self.tape.cross_entropy(logits, target.clone())?;
            total = Some(match total {
                Some(previous) => self.tape.add(previous, sequence_loss)?,
                None => sequence_loss,
            });
        }
        Ok(self.tape.scale(
            total.expect("validated non-empty batch"),
            1.0 / tokens.len() as f32,
        )?)
    }

    pub fn loss_value(&self, loss: TensorId) -> Result<f32, GptError> {
        Ok(self.tape.value(loss)?.data()[0])
    }

    pub fn backward(
        &self,
        loss: TensorId,
    ) -> Result<std::collections::HashMap<TensorId, Tensor>, GptError> {
        Ok(self.tape.backward(loss)?)
    }

    pub(crate) fn tape_mut(&mut self) -> &mut Tape {
        &mut self.tape
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<(), GptError> {
        let mut file = File::create(path)?;
        file.write_all(b"GBX1")?;
        let values = self.parameter_values()?;
        write_u64(&mut file, values.len() as u64)?;
        for tensor in values {
            write_u64(&mut file, tensor.shape().len() as u64)?;
            for &dimension in tensor.shape() {
                write_u64(&mut file, dimension as u64)?;
            }
            write_u64(&mut file, tensor.data().len() as u64)?;
            for &value in tensor.data() {
                file.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: impl AsRef<Path>) -> Result<(), GptError> {
        self.reset_tape()?;
        let mut file = File::open(path)?;
        let mut magic = [0_u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GBX1" {
            return Err(GptError::InvalidCheckpoint("bad magic".to_string()));
        }
        let parameters = self.parameters();
        let count = read_u64(&mut file)? as usize;
        if count != parameters.len() {
            return Err(GptError::InvalidCheckpoint(format!(
                "expected {} parameters, found {count}",
                parameters.len()
            )));
        }
        let mut tensors = Vec::with_capacity(count);
        for &id in &parameters {
            let rank = read_u64(&mut file)? as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(read_u64(&mut file)? as usize);
            }
            let length = read_u64(&mut file)? as usize;
            let mut data = Vec::with_capacity(length);
            for _ in 0..length {
                let mut bytes = [0_u8; 4];
                file.read_exact(&mut bytes)?;
                data.push(f32::from_le_bytes(bytes));
            }
            let expected = self.tape.value(id)?.shape();
            if shape != expected {
                return Err(GptError::InvalidCheckpoint(format!(
                    "parameter {id} has shape {shape:?}, expected {expected:?}"
                )));
            }
            tensors.push(Tensor::from_vec(shape, data).map_err(TapeError::from)?);
        }
        for (id, tensor) in parameters.into_iter().zip(tensors) {
            self.tape.overwrite_leaf(id, tensor)?;
        }
        Ok(())
    }
}
fn write_u64(file: &mut File, value: u64) -> Result<(), std::io::Error> {
    file.write_all(&value.to_le_bytes())
}

fn read_u64(file: &mut File) -> Result<u64, std::io::Error> {
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
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

    #[test]
    fn gpt_forward_has_batch_sequence_vocab_shape() {
        let mut model = Gpt::new(65, 16, 2, 1, 8).unwrap();
        let tokens = vec![
            (0..16).map(|index| (index * 7) % 65).collect(),
            (0..16).map(|index| (index * 11 + 3) % 65).collect(),
        ];
        let logits = model.forward(&tokens).unwrap();
        assert_eq!(logits.shape(), &[2, 16, 65]);
        assert!(logits.data().iter().all(|value| value.is_finite()));
        assert_eq!(model.parameters().len(), 38);
    }

    #[test]
    fn gpt_rejects_unsupported_heads_and_invalid_batches() {
        assert!(matches!(
            Gpt::new(65, 16, 1, 2, 8),
            Err(GptError::UnsupportedHeadCount(2))
        ));
        let mut model = Gpt::new(65, 16, 1, 1, 8).unwrap();
        assert!(matches!(model.forward(&[]), Err(GptError::EmptyBatch)));
        assert!(matches!(
            model.forward(&[vec![1, 2], vec![3]]),
            Err(GptError::RaggedBatch)
        ));
        assert!(matches!(
            model.forward(&[vec![0; 17]]),
            Err(GptError::SequenceTooLong { .. })
        ));
    }

    #[test]
    fn gpt_loss_stays_on_tape_and_reaches_parameters() {
        let mut model = Gpt::new_with_seed(65, 4, 1, 1, 8, 42).unwrap();
        let tokens = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let targets = vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9]];
        let loss = model.loss(&tokens, &targets).unwrap();
        assert!(model.tape.value(loss).unwrap().shape().is_empty());
        assert!((model.loss_value(loss).unwrap() - 65.0_f32.ln()).abs() < 0.1);
        let gradients = model.backward(loss).unwrap();
        let parameters = model.parameters();
        assert!(parameters.iter().all(|id| gradients.contains_key(id)));
        assert!(
            parameters
                .iter()
                .flat_map(|id| gradients[id].data())
                .any(|value| value.abs() > 1e-8)
        );
    }

    #[test]
    fn gpt_reset_keeps_parameter_ids_and_values_without_growth() {
        let mut model = Gpt::new_with_seed(17, 4, 1, 1, 8, 9).unwrap();
        let parameters = model.parameters();
        let values = model.parameter_values().unwrap();
        let checkpoint = model.tape_entry_count();
        let tokens = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let targets = vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9]];
        for _ in 0..5 {
            model.loss(&tokens, &targets).unwrap();
            assert!(model.tape_entry_count() > checkpoint);
            model.reset_tape().unwrap();
            assert_eq!(model.tape_entry_count(), checkpoint);
            assert_eq!(model.parameters(), parameters);
            assert_eq!(model.parameter_values().unwrap(), values);
        }
    }

    #[test]
    fn checkpoint_round_trip_preserves_logits() {
        let tokens = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let mut source = Gpt::new_with_seed(17, 4, 1, 1, 8, 11).unwrap();
        let expected = source.forward(&tokens).unwrap();
        let path =
            std::env::temp_dir().join(format!("glassbox-checkpoint-{}.bin", std::process::id()));
        source.save_checkpoint(&path).unwrap();

        let mut loaded = Gpt::new_with_seed(17, 4, 1, 1, 8, 99).unwrap();
        loaded.load_checkpoint(&path).unwrap();
        let actual = loaded.forward(&tokens).unwrap();
        assert_eq!(actual, expected);
        std::fs::remove_file(path).unwrap();
    }
}
