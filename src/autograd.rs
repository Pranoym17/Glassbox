use crate::visualizer::{Event, EventEmitter, Phase};
use crate::{
    Tensor, TensorError,
    transformer::{LayerNormContext, layer_norm, layer_norm_backward, softmax, softmax_backward},
};
use std::collections::{HashMap, HashSet};
use std::fmt;
pub type TensorId = usize;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpType {
    Leaf,
    Add,
    Mul,
    Sub,
    Div,
    Exp,
    Log,
    MatMul,
    Transpose,
    Scale,
    CausalMask,
    Softmax,
    Embedding,
    LayerNorm,
    Gelu,
    Relu,
    Sigmoid,
    Tanh,
    Reshape,
    CrossEntropy,
}
impl OpType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Sub => "sub",
            Self::Div => "div",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::MatMul => "matmul",
            Self::Transpose => "transpose",
            Self::Scale => "scale",
            Self::CausalMask => "causal_mask",
            Self::Softmax => "softmax",
            Self::Embedding => "embedding",
            Self::LayerNorm => "layer_norm",
            Self::Gelu => "gelu",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Reshape => "reshape",
            Self::CrossEntropy => "cross_entropy",
        }
    }
}
#[derive(Clone, Debug)]
pub enum SavedContext {
    None,
    Binary {
        left: Tensor,
        right: Tensor,
    },
    Unary {
        input: Tensor,
        output: Tensor,
    },
    Scale {
        factor: f32,
    },
    Softmax {
        output: Tensor,
    },
    Embedding {
        indices: Vec<usize>,
        table_shape: Vec<usize>,
    },
    LayerNorm {
        context: LayerNormContext,
        gamma: Tensor,
        epsilon: f32,
    },
    Gelu {
        input: Tensor,
    },
    Activation {
        output: Tensor,
    },
    Reshape {
        input_shape: Vec<usize>,
    },
    CrossEntropy {
        probabilities: Tensor,
        targets: Vec<usize>,
    },
}
#[derive(Clone, Debug)]
pub struct TapeEntry {
    pub id: TensorId,
    pub op: OpType,
    pub inputs: Vec<TensorId>,
    pub saved: SavedContext,
}
#[derive(Debug)]
pub enum TapeError {
    UnknownTensor(TensorId),
    Tensor(TensorError),
    NonLeafUpdate(TensorId),
    InvalidResetPoint(usize),
}
impl fmt::Display for TapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTensor(id) => write!(f, "unknown tensor {id}"),
            Self::Tensor(e) => write!(f, "tensor error: {e}"),
            Self::NonLeafUpdate(id) => write!(f, "tensor {id} is not a leaf"),
            Self::InvalidResetPoint(checkpoint) => {
                write!(
                    f,
                    "tape checkpoint {checkpoint} does not contain only existing leaves"
                )
            }
        }
    }
}
impl std::error::Error for TapeError {}
impl From<TensorError> for TapeError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}
pub struct Tape {
    entries: Vec<TapeEntry>,
    index: HashMap<TensorId, usize>,
    values: HashMap<TensorId, Tensor>,
    next_id: TensorId,
    emitter: Option<EventEmitter>,
    trace_enabled: bool,
    trace_step: u64,
}
impl Tape {
    pub fn new() -> Self {
        Self {
            entries: vec![],
            index: HashMap::new(),
            values: HashMap::new(),
            next_id: 0,
            emitter: None,
            trace_enabled: false,
            trace_step: 0,
        }
    }
    pub fn leaf(&mut self, v: Tensor) -> TensorId {
        self.append(OpType::Leaf, vec![], SavedContext::None, v)
    }
    pub fn value(&self, id: TensorId) -> Result<&Tensor, TapeError> {
        self.values.get(&id).ok_or(TapeError::UnknownTensor(id))
    }
    pub fn entry(&self, id: TensorId) -> Result<&TapeEntry, TapeError> {
        Ok(&self.entries[*self.index.get(&id).ok_or(TapeError::UnknownTensor(id))?])
    }
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
    pub fn set_event_emitter(&mut self, emitter: Option<EventEmitter>) {
        self.emitter = emitter;
    }
    pub fn set_trace(&mut self, step: u64, enabled: bool) {
        self.trace_step = step;
        self.trace_enabled = enabled;
    }
    pub fn overwrite_leaf(&mut self, id: TensorId, value: Tensor) -> Result<(), TapeError> {
        if self.entry(id)?.op != OpType::Leaf {
            return Err(TapeError::NonLeafUpdate(id));
        }
        let current_shape = self.value(id)?.shape().to_vec();
        if current_shape != value.shape() {
            return Err(TensorError::IncompatibleShapes {
                left: current_shape,
                right: value.shape().to_vec(),
            }
            .into());
        }
        self.values.insert(id, value);
        Ok(())
    }
    pub fn reset(&mut self, checkpoint: usize) -> Result<(), TapeError> {
        if checkpoint > self.entries.len()
            || self.entries[..checkpoint]
                .iter()
                .any(|entry| entry.op != OpType::Leaf)
        {
            return Err(TapeError::InvalidResetPoint(checkpoint));
        }
        let removed: Vec<_> = self.entries[checkpoint..]
            .iter()
            .map(|entry| entry.id)
            .collect();
        self.entries.truncate(checkpoint);
        for id in removed {
            self.values.remove(&id);
        }
        self.index.clear();
        for (position, entry) in self.entries.iter().enumerate() {
            self.index.insert(entry.id, position);
        }
        self.next_id = self
            .entries
            .iter()
            .map(|entry| entry.id + 1)
            .max()
            .unwrap_or(0);
        Ok(())
    }
    pub fn retain_leaves(&mut self, retained: &[TensorId]) -> Result<(), TapeError> {
        let mut seen = HashSet::new();
        let mut entries = Vec::with_capacity(retained.len());
        let mut index = HashMap::with_capacity(retained.len());
        let mut values = HashMap::with_capacity(retained.len());
        for &id in retained {
            if !seen.insert(id) {
                continue;
            }
            let entry = self.entry(id)?.clone();
            if entry.op != OpType::Leaf {
                return Err(TapeError::NonLeafUpdate(id));
            }
            let value = self.value(id)?.clone();
            index.insert(id, entries.len());
            entries.push(entry);
            values.insert(id, value);
        }
        self.entries = entries;
        self.index = index;
        self.values = values;
        Ok(())
    }
    /// Creates a new leaf that shares the value but has no edge to the source graph.
    pub fn detach(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let value = self.value(input)?.detach();
        Ok(self.leaf(value))
    }
    pub fn add(&mut self, l: TensorId, r: TensorId) -> Result<TensorId, TapeError> {
        let (a, b) = (self.value(l)?.clone(), self.value(r)?.clone());
        let out = a.add(&b)?;
        Ok(self.append(
            OpType::Add,
            vec![l, r],
            SavedContext::Binary { left: a, right: b },
            out,
        ))
    }
    pub fn mul(&mut self, l: TensorId, r: TensorId) -> Result<TensorId, TapeError> {
        let (a, b) = (self.value(l)?.clone(), self.value(r)?.clone());
        let out = a.mul(&b)?;
        Ok(self.append(
            OpType::Mul,
            vec![l, r],
            SavedContext::Binary { left: a, right: b },
            out,
        ))
    }
    pub fn sub(&mut self, l: TensorId, r: TensorId) -> Result<TensorId, TapeError> {
        let (left, right) = (self.value(l)?.clone(), self.value(r)?.clone());
        let output = left.sub(&right)?;
        Ok(self.append(
            OpType::Sub,
            vec![l, r],
            SavedContext::Binary { left, right },
            output,
        ))
    }
    pub fn div(&mut self, l: TensorId, r: TensorId) -> Result<TensorId, TapeError> {
        let (left, right) = (self.value(l)?.clone(), self.value(r)?.clone());
        let output = left.div(&right)?;
        Ok(self.append(
            OpType::Div,
            vec![l, r],
            SavedContext::Binary { left, right },
            output,
        ))
    }
    pub fn exp(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let value = self.value(input)?.clone();
        let output = value.exp()?;
        Ok(self.append(
            OpType::Exp,
            vec![input],
            SavedContext::Unary {
                input: value,
                output: output.clone(),
            },
            output,
        ))
    }
    pub fn log(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let value = self.value(input)?.clone();
        let output = value.log()?;
        Ok(self.append(
            OpType::Log,
            vec![input],
            SavedContext::Unary {
                input: value,
                output: output.clone(),
            },
            output,
        ))
    }
    pub fn matmul(&mut self, l: TensorId, r: TensorId) -> Result<TensorId, TapeError> {
        let (a, b) = (self.value(l)?.clone(), self.value(r)?.clone());
        let o = a.matmul(&b)?;
        Ok(self.append(
            OpType::MatMul,
            vec![l, r],
            SavedContext::Binary { left: a, right: b },
            o,
        ))
    }
    pub fn transpose(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let output = self.value(input)?.permute(&[1, 0])?;
        Ok(self.append(OpType::Transpose, vec![input], SavedContext::None, output))
    }
    pub fn scale(&mut self, input: TensorId, factor: f32) -> Result<TensorId, TapeError> {
        let scalar = Tensor::from_vec(vec![], vec![factor])?;
        let output = self.value(input)?.mul(&scalar)?;
        Ok(self.append(
            OpType::Scale,
            vec![input],
            SavedContext::Scale { factor },
            output,
        ))
    }
    pub fn causal_mask(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let value = self.value(input)?;
        if value.shape().len() != 2 || value.shape()[0] != value.shape()[1] {
            return Err(TensorError::IncompatibleShapes {
                left: value.shape().to_vec(),
                right: vec![value.shape()[0], value.shape()[0]],
            }
            .into());
        }
        let size = value.shape()[0];
        let mut data = value.contiguous().data().to_vec();
        for row in 0..size {
            for column in row + 1..size {
                data[row * size + column] = -1e9;
            }
        }
        let output = Tensor::from_vec(vec![size, size], data)?;
        Ok(self.append(OpType::CausalMask, vec![input], SavedContext::None, output))
    }
    pub fn softmax(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let output = softmax(self.value(input)?)?;
        Ok(self.append(
            OpType::Softmax,
            vec![input],
            SavedContext::Softmax {
                output: output.clone(),
            },
            output,
        ))
    }
    pub fn embedding(
        &mut self,
        table: TensorId,
        indices: Vec<usize>,
    ) -> Result<TensorId, TapeError> {
        let value = self.value(table)?.contiguous();
        if value.shape().len() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: value.shape().len(),
            }
            .into());
        }
        let (rows, columns) = (value.shape()[0], value.shape()[1]);
        let mut data = Vec::with_capacity(indices.len() * columns);
        for &index in &indices {
            if index >= rows {
                return Err(TensorError::IndexOutOfBounds {
                    axis: 0,
                    index,
                    dimension: rows,
                }
                .into());
            }
            data.extend_from_slice(&value.data()[index * columns..(index + 1) * columns]);
        }
        let output = Tensor::from_vec(vec![indices.len(), columns], data)?;
        Ok(self.append(
            OpType::Embedding,
            vec![table],
            SavedContext::Embedding {
                indices,
                table_shape: value.shape().to_vec(),
            },
            output,
        ))
    }
    pub fn layer_norm(
        &mut self,
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        epsilon: f32,
    ) -> Result<TensorId, TapeError> {
        let gamma_value = self.value(gamma)?.clone();
        let (output, context) =
            layer_norm(self.value(input)?, &gamma_value, self.value(beta)?, epsilon)?;
        Ok(self.append(
            OpType::LayerNorm,
            vec![input, gamma, beta],
            SavedContext::LayerNorm {
                context,
                gamma: gamma_value,
                epsilon,
            },
            output,
        ))
    }
    pub fn gelu(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let value = self.value(input)?.contiguous();
        let output = Tensor::from_vec(
            value.shape().to_vec(),
            value.data().iter().copied().map(gelu_value).collect(),
        )?;
        Ok(self.append(
            OpType::Gelu,
            vec![input],
            SavedContext::Gelu { input: value },
            output,
        ))
    }
    pub fn relu(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let output = self.value(input)?.relu()?;
        Ok(self.append(
            OpType::Relu,
            vec![input],
            SavedContext::Activation {
                output: output.clone(),
            },
            output,
        ))
    }
    pub fn sigmoid(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let output = self.value(input)?.sigmoid()?;
        Ok(self.append(
            OpType::Sigmoid,
            vec![input],
            SavedContext::Activation {
                output: output.clone(),
            },
            output,
        ))
    }
    pub fn tanh(&mut self, input: TensorId) -> Result<TensorId, TapeError> {
        let output = self.value(input)?.tanh()?;
        Ok(self.append(
            OpType::Tanh,
            vec![input],
            SavedContext::Activation {
                output: output.clone(),
            },
            output,
        ))
    }
    pub fn reshape(&mut self, input: TensorId, shape: Vec<usize>) -> Result<TensorId, TapeError> {
        let input_shape = self.value(input)?.shape().to_vec();
        let output = self.value(input)?.reshape(shape)?;
        Ok(self.append(
            OpType::Reshape,
            vec![input],
            SavedContext::Reshape { input_shape },
            output,
        ))
    }
    pub fn cross_entropy(
        &mut self,
        logits: TensorId,
        targets: Vec<usize>,
    ) -> Result<TensorId, TapeError> {
        let values = self.value(logits)?.contiguous();
        if values.shape().len() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: values.shape().len(),
            }
            .into());
        }
        let (rows, columns) = (values.shape()[0], values.shape()[1]);
        if targets.len() != rows {
            return Err(TensorError::DataLengthMismatch {
                expected: rows,
                actual: targets.len(),
            }
            .into());
        }
        let mut probabilities = vec![0.0; rows * columns];
        let mut loss = 0.0;
        for (row, &target) in targets.iter().enumerate() {
            if target >= columns {
                return Err(TensorError::IndexOutOfBounds {
                    axis: 1,
                    index: target,
                    dimension: columns,
                }
                .into());
            }
            let values = &values.data()[row * columns..(row + 1) * columns];
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = values.iter().map(|value| (value - maximum).exp()).sum();
            loss += maximum + sum.ln() - values[target];
            for column in 0..columns {
                probabilities[row * columns + column] = (values[column] - maximum).exp() / sum;
            }
        }
        let probabilities = Tensor::from_vec(vec![rows, columns], probabilities)?;
        let output = Tensor::from_vec(vec![], vec![loss / rows as f32])?;
        Ok(self.append(
            OpType::CrossEntropy,
            vec![logits],
            SavedContext::CrossEntropy {
                probabilities,
                targets,
            },
            output,
        ))
    }
    pub fn backward(&self, loss: TensorId) -> Result<HashMap<TensorId, Tensor>, TapeError> {
        let mut seen = HashSet::new();
        let mut order = vec![];
        self.visit(loss, &mut seen, &mut order)?;
        let mut g = HashMap::new();
        g.insert(loss, Tensor::ones(self.value(loss)?.shape().to_vec())?);
        for id in order.into_iter().rev() {
            let Some(up) = g.get(&id).cloned() else {
                continue;
            };
            let e = self.entry(id)?;
            match (&e.op, &e.saved) {
                (OpType::Leaf, _) => {}
                (OpType::Add, _) => {
                    self.acc(
                        &mut g,
                        e.inputs[0],
                        up.sum_to_shape(self.value(e.inputs[0])?.shape())?,
                    )?;
                    self.acc(
                        &mut g,
                        e.inputs[1],
                        up.sum_to_shape(self.value(e.inputs[1])?.shape())?,
                    )?
                }
                (OpType::Mul, SavedContext::Binary { left, right }) => {
                    self.acc(
                        &mut g,
                        e.inputs[0],
                        up.mul(right)?
                            .sum_to_shape(self.value(e.inputs[0])?.shape())?,
                    )?;
                    self.acc(
                        &mut g,
                        e.inputs[1],
                        up.mul(left)?
                            .sum_to_shape(self.value(e.inputs[1])?.shape())?,
                    )?
                }
                (OpType::Sub, _) => {
                    self.acc(
                        &mut g,
                        e.inputs[0],
                        up.sum_to_shape(self.value(e.inputs[0])?.shape())?,
                    )?;
                    self.acc(
                        &mut g,
                        e.inputs[1],
                        up.mul(&Tensor::from_vec(vec![], vec![-1.0])?)?
                            .sum_to_shape(self.value(e.inputs[1])?.shape())?,
                    )?
                }
                (OpType::Div, SavedContext::Binary { left, right }) => {
                    self.acc(
                        &mut g,
                        e.inputs[0],
                        up.div(right)?
                            .sum_to_shape(self.value(e.inputs[0])?.shape())?,
                    )?;
                    let denominator = right.mul(right)?;
                    let negative = Tensor::from_vec(vec![], vec![-1.0])?;
                    self.acc(
                        &mut g,
                        e.inputs[1],
                        up.mul(left)?
                            .div(&denominator)?
                            .mul(&negative)?
                            .sum_to_shape(self.value(e.inputs[1])?.shape())?,
                    )?
                }
                (OpType::Exp, SavedContext::Unary { output, .. }) => {
                    self.acc(&mut g, e.inputs[0], up.mul(output)?)?
                }
                (OpType::Log, SavedContext::Unary { input, .. }) => {
                    self.acc(&mut g, e.inputs[0], up.div(input)?)?
                }
                (OpType::MatMul, SavedContext::Binary { left, right }) => {
                    let rt = right.permute(&[1, 0])?;
                    let lt = left.permute(&[1, 0])?;
                    self.acc(&mut g, e.inputs[0], up.matmul(&rt)?)?;
                    self.acc(&mut g, e.inputs[1], lt.matmul(&up)?)?
                }
                (OpType::Transpose, _) => {
                    self.acc(&mut g, e.inputs[0], up.permute(&[1, 0])?.contiguous())?
                }
                (OpType::Scale, SavedContext::Scale { factor }) => {
                    let scalar = Tensor::from_vec(vec![], vec![*factor])?;
                    self.acc(&mut g, e.inputs[0], up.mul(&scalar)?)?
                }
                (OpType::CausalMask, _) => {
                    let size = up.shape()[0];
                    let mut data = up.contiguous().data().to_vec();
                    for row in 0..size {
                        for column in row + 1..size {
                            data[row * size + column] = 0.0;
                        }
                    }
                    self.acc(
                        &mut g,
                        e.inputs[0],
                        Tensor::from_vec(vec![size, size], data)?,
                    )?
                }
                (OpType::Softmax, SavedContext::Softmax { output }) => {
                    self.acc(&mut g, e.inputs[0], softmax_backward(output, &up)?)?
                }
                (
                    OpType::Embedding,
                    SavedContext::Embedding {
                        indices,
                        table_shape,
                    },
                ) => {
                    let columns = table_shape[1];
                    let mut data = vec![0.0; table_shape[0] * columns];
                    let upstream = up.contiguous();
                    for (row, &index) in indices.iter().enumerate() {
                        for column in 0..columns {
                            data[index * columns + column] +=
                                upstream.data()[row * columns + column];
                        }
                    }
                    self.acc(
                        &mut g,
                        e.inputs[0],
                        Tensor::from_vec(table_shape.clone(), data)?,
                    )?
                }
                (
                    OpType::LayerNorm,
                    SavedContext::LayerNorm {
                        context,
                        gamma,
                        epsilon,
                    },
                ) => {
                    let (dx, dgamma, dbeta) = layer_norm_backward(&up, context, gamma, *epsilon)?;
                    self.acc(&mut g, e.inputs[0], dx)?;
                    self.acc(&mut g, e.inputs[1], dgamma)?;
                    self.acc(&mut g, e.inputs[2], dbeta)?
                }
                (OpType::Gelu, SavedContext::Gelu { input }) => {
                    let derivative = Tensor::from_vec(
                        input.shape().to_vec(),
                        input.data().iter().copied().map(gelu_derivative).collect(),
                    )?;
                    self.acc(&mut g, e.inputs[0], up.mul(&derivative)?)?
                }
                (OpType::Relu, SavedContext::Activation { output }) => {
                    let derivative = Tensor::from_vec(
                        output.shape().to_vec(),
                        output
                            .data()
                            .iter()
                            .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
                            .collect(),
                    )?;
                    self.acc(&mut g, e.inputs[0], up.mul(&derivative)?)?
                }
                (OpType::Sigmoid, SavedContext::Activation { output }) => {
                    let derivative = Tensor::from_vec(
                        output.shape().to_vec(),
                        output
                            .data()
                            .iter()
                            .map(|value| value * (1.0 - value))
                            .collect(),
                    )?;
                    self.acc(&mut g, e.inputs[0], up.mul(&derivative)?)?
                }
                (OpType::Tanh, SavedContext::Activation { output }) => {
                    let derivative = Tensor::from_vec(
                        output.shape().to_vec(),
                        output
                            .data()
                            .iter()
                            .map(|value| 1.0 - value * value)
                            .collect(),
                    )?;
                    self.acc(&mut g, e.inputs[0], up.mul(&derivative)?)?
                }
                (OpType::Reshape, SavedContext::Reshape { input_shape }) => {
                    self.acc(&mut g, e.inputs[0], up.reshape(input_shape.clone())?)?
                }
                (
                    OpType::CrossEntropy,
                    SavedContext::CrossEntropy {
                        probabilities,
                        targets,
                    },
                ) => {
                    let rows = probabilities.shape()[0];
                    let columns = probabilities.shape()[1];
                    let factor = up.data()[0] / rows as f32;
                    let mut data = probabilities.data().to_vec();
                    for (row, &target) in targets.iter().enumerate() {
                        data[row * columns + target] -= 1.0;
                    }
                    for value in &mut data {
                        *value *= factor;
                    }
                    self.acc(
                        &mut g,
                        e.inputs[0],
                        Tensor::from_vec(probabilities.shape().to_vec(), data)?,
                    )?
                }
                _ => unreachable!(),
            }
            self.emit_event(Phase::Backward, id, Some(l2_norm(&up)));
        }
        Ok(g)
    }
    fn append(
        &mut self,
        op: OpType,
        inputs: Vec<TensorId>,
        saved: SavedContext,
        value: Tensor,
    ) -> TensorId {
        let id = self.next_id;
        self.next_id += 1;
        self.index.insert(id, self.entries.len());
        self.entries.push(TapeEntry {
            id,
            op,
            inputs,
            saved,
        });
        self.values.insert(id, value);
        self.emit_event(Phase::Forward, id, None);
        id
    }
    fn emit_event(&self, phase: Phase, id: TensorId, grad_norm: Option<f32>) {
        if !self.trace_enabled {
            return;
        }
        let Some(emitter) = &self.emitter else {
            return;
        };
        let entry = &self.entries[self.index[&id]];
        let value = &self.values[&id];
        emitter.emit(Event {
            step: self.trace_step,
            phase,
            op: entry.op.as_str().to_string(),
            output_id: id,
            input_ids: entry.inputs.clone(),
            shape: value.shape().to_vec(),
            grad_norm,
        });
    }
    fn visit(
        &self,
        id: TensorId,
        s: &mut HashSet<TensorId>,
        o: &mut Vec<TensorId>,
    ) -> Result<(), TapeError> {
        if !s.insert(id) {
            return Ok(());
        }
        for &i in &self.entry(id)?.inputs {
            self.visit(i, s, o)?
        }
        o.push(id);
        Ok(())
    }
    fn acc(
        &self,
        g: &mut HashMap<TensorId, Tensor>,
        id: TensorId,
        x: Tensor,
    ) -> Result<(), TapeError> {
        let n = if let Some(old) = g.get(&id) {
            old.add(&x)?
        } else {
            x
        };
        g.insert(id, n);
        Ok(())
    }
}
impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}
fn l2_norm(tensor: &Tensor) -> f32 {
    tensor
        .data()
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}
fn gelu_value(x: f32) -> f32 {
    let u = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
    0.5 * x * (1.0 + u.tanh())
}
fn gelu_derivative(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    let u = c * (x + 0.044715 * x.powi(3));
    let tanh = u.tanh();
    0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * c * (1.0 + 3.0 * 0.044715 * x * x)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    #[test]
    fn three_op_graph() {
        let mut t = Tape::new();
        let a = t.leaf(Tensor::from_vec(vec![], vec![2.]).unwrap());
        let b = t.leaf(Tensor::from_vec(vec![], vec![3.]).unwrap());
        let c = t.leaf(Tensor::from_vec(vec![], vec![4.]).unwrap());
        let p = t.mul(a, b).unwrap();
        let y = t.add(p, c).unwrap();
        assert_eq!(t.entry(p).unwrap().op, OpType::Mul);
        let g = t.backward(y).unwrap();
        assert_eq!(g[&a].data(), &[3.]);
        assert_eq!(g[&b].data(), &[2.]);
        assert_eq!(g[&c].data(), &[1.]);
    }

    fn objective(a: &Tensor, b: &Tensor, c: &Tensor, w: &Tensor) -> f32 {
        a.matmul(b)
            .unwrap()
            .add(c)
            .unwrap()
            .mul(w)
            .unwrap()
            .data()
            .iter()
            .sum()
    }

    fn check_gradient(analytic: &Tensor, values: &Tensor, evaluate: impl Fn(Tensor) -> f32) {
        let eps = 1e-3;
        for index in 0..values.data().len() {
            let mut plus = values.data().to_vec();
            let mut minus = plus.clone();
            plus[index] += eps;
            minus[index] -= eps;
            let plus = Tensor::from_vec(values.shape().to_vec(), plus).unwrap();
            let minus = Tensor::from_vec(values.shape().to_vec(), minus).unwrap();
            let numerical = (evaluate(plus) - evaluate(minus)) / (2.0 * eps);
            assert!(
                (analytic.data()[index] - numerical).abs() < 5e-3,
                "gradient {index}: analytic={}, numerical={numerical}",
                analytic.data()[index]
            );
        }
    }

    #[test]
    fn matmul_add_mul_backward_matches_finite_differences() {
        let a_value = Tensor::from_vec(vec![2, 3], vec![0.2, -0.4, 0.7, 1.1, 0.3, -0.2]).unwrap();
        let b_value = Tensor::from_vec(vec![3, 2], vec![0.5, -0.3, 0.8, 0.2, -0.6, 0.9]).unwrap();
        let c_value = Tensor::from_vec(vec![2, 2], vec![0.1, -0.2, 0.4, 0.3]).unwrap();
        let w_value = Tensor::from_vec(vec![2, 2], vec![0.7, -0.5, 1.2, 0.4]).unwrap();

        let mut tape = Tape::new();
        let a = tape.leaf(a_value.clone());
        let b = tape.leaf(b_value.clone());
        let c = tape.leaf(c_value.clone());
        let w = tape.leaf(w_value.clone());
        let product = tape.matmul(a, b).unwrap();
        let shifted = tape.add(product, c).unwrap();
        let loss = tape.mul(shifted, w).unwrap();
        let gradients = tape.backward(loss).unwrap();

        check_gradient(&gradients[&a], &a_value, |value| {
            objective(&value, &b_value, &c_value, &w_value)
        });
        check_gradient(&gradients[&b], &b_value, |value| {
            objective(&a_value, &value, &c_value, &w_value)
        });
        check_gradient(&gradients[&c], &c_value, |value| {
            objective(&a_value, &b_value, &value, &w_value)
        });
        check_gradient(&gradients[&w], &w_value, |value| {
            objective(&a_value, &b_value, &c_value, &value)
        });
    }

    fn masked_softmax_objective(input: &Tensor, weights: &Tensor) -> f32 {
        let size = input.shape()[0];
        let scalar = Tensor::from_vec(vec![], vec![0.7]).unwrap();
        let scaled = input.mul(&scalar).unwrap();
        let mut data = scaled.data().to_vec();
        for row in 0..size {
            for column in row + 1..size {
                data[row * size + column] = -1e9;
            }
        }
        softmax(&Tensor::from_vec(vec![size, size], data).unwrap())
            .unwrap()
            .mul(weights)
            .unwrap()
            .data()
            .iter()
            .sum()
    }

    #[test]
    fn softmax_scale_mask_backward_matches_finite_differences() {
        let input = Tensor::from_vec(
            vec![3, 3],
            vec![0.2, -0.4, 0.7, 1.1, 0.3, -0.2, -0.5, 0.8, 0.4],
        )
        .unwrap();
        let weights = Tensor::from_vec(
            vec![3, 3],
            vec![0.3, -0.6, 0.2, 0.7, -0.1, 0.5, -0.4, 0.9, 0.1],
        )
        .unwrap();
        let mut tape = Tape::new();
        let input_id = tape.leaf(input.clone());
        let weights_id = tape.leaf(weights.clone());
        let scaled = tape.scale(input_id, 0.7).unwrap();
        let masked = tape.causal_mask(scaled).unwrap();
        let probabilities = tape.softmax(masked).unwrap();
        let loss = tape.mul(probabilities, weights_id).unwrap();
        let gradients = tape.backward(loss).unwrap();

        check_gradient(&gradients[&input_id], &input, |value| {
            masked_softmax_objective(&value, &weights)
        });
    }

    #[test]
    fn sub_backward_matches_finite_differences_with_broadcasting() {
        let left = Tensor::from_vec(vec![2, 2], vec![0.4, -0.7, 1.3, 0.2]).unwrap();
        let right = Tensor::from_vec(vec![2], vec![0.6, -0.3]).unwrap();
        let mut tape = Tape::new();
        let left_id = tape.leaf(left.clone());
        let right_id = tape.leaf(right.clone());
        let output = tape.sub(left_id, right_id).unwrap();
        let gradients = tape.backward(output).unwrap();
        check_gradient(&gradients[&left_id], &left, |value| {
            value.sub(&right).unwrap().data().iter().sum()
        });
        check_gradient(&gradients[&right_id], &right, |value| {
            left.sub(&value).unwrap().data().iter().sum()
        });
    }

    #[test]
    fn div_backward_matches_finite_differences_with_broadcasting() {
        let left = Tensor::from_vec(vec![2, 2], vec![0.4, -0.7, 1.3, 0.2]).unwrap();
        let right = Tensor::from_vec(vec![2], vec![0.6, -1.3]).unwrap();
        let mut tape = Tape::new();
        let left_id = tape.leaf(left.clone());
        let right_id = tape.leaf(right.clone());
        let output = tape.div(left_id, right_id).unwrap();
        let gradients = tape.backward(output).unwrap();
        check_gradient(&gradients[&left_id], &left, |value| {
            value.div(&right).unwrap().data().iter().sum()
        });
        check_gradient(&gradients[&right_id], &right, |value| {
            left.div(&value).unwrap().data().iter().sum()
        });
    }

    #[test]
    fn exp_backward_matches_finite_differences() {
        let values = Tensor::from_vec(vec![4], vec![-1.2, -0.4, 0.3, 1.5]).unwrap();
        let mut tape = Tape::new();
        let input = tape.leaf(values.clone());
        let output = tape.exp(input).unwrap();
        let gradients = tape.backward(output).unwrap();
        check_gradient(&gradients[&input], &values, |value| {
            value.exp().unwrap().data().iter().sum()
        });
    }

    #[test]
    fn log_backward_matches_finite_differences() {
        let values = Tensor::from_vec(vec![4], vec![0.3, 0.8, 1.7, 3.2]).unwrap();
        let mut tape = Tape::new();
        let input = tape.leaf(values.clone());
        let output = tape.log(input).unwrap();
        let gradients = tape.backward(output).unwrap();
        check_gradient(&gradients[&input], &values, |value| {
            value.log().unwrap().data().iter().sum()
        });
    }

    #[test]
    fn individual_transformer_tape_ops_match_finite_differences() {
        let input = Tensor::from_vec(vec![2, 2], vec![0.2, -0.4, 0.7, 0.1]).unwrap();
        let weights = Tensor::from_vec(vec![2, 2], vec![0.3, -0.6, 0.2, 0.8]).unwrap();

        let mut scale_tape = Tape::new();
        let scale_input = scale_tape.leaf(input.clone());
        let scale_weights = scale_tape.leaf(weights.clone());
        let scaled = scale_tape.scale(scale_input, 0.7).unwrap();
        let scale_loss = scale_tape.mul(scaled, scale_weights).unwrap();
        let scale_gradients = scale_tape.backward(scale_loss).unwrap();
        check_gradient(&scale_gradients[&scale_input], &input, |value| {
            value
                .mul(&Tensor::from_vec(vec![], vec![0.7]).unwrap())
                .unwrap()
                .mul(&weights)
                .unwrap()
                .data()
                .iter()
                .sum()
        });

        let mut softmax_tape = Tape::new();
        let softmax_input = softmax_tape.leaf(input.clone());
        let softmax_weights = softmax_tape.leaf(weights.clone());
        let probabilities = softmax_tape.softmax(softmax_input).unwrap();
        let softmax_loss = softmax_tape.mul(probabilities, softmax_weights).unwrap();
        let softmax_gradients = softmax_tape.backward(softmax_loss).unwrap();
        check_gradient(&softmax_gradients[&softmax_input], &input, |value| {
            softmax(&value)
                .unwrap()
                .mul(&weights)
                .unwrap()
                .data()
                .iter()
                .sum()
        });

        let mask_input = Tensor::from_vec(
            vec![3, 3],
            vec![0.2, -0.4, 0.7, 1.1, 0.3, -0.2, -0.5, 0.8, 0.4],
        )
        .unwrap();
        let mask_weights = Tensor::from_vec(
            vec![3, 3],
            vec![0.3, 0.0, 0.0, 0.7, -0.1, 0.0, -0.4, 0.9, 0.1],
        )
        .unwrap();
        let mut mask_tape = Tape::new();
        let mask_input_id = mask_tape.leaf(mask_input.clone());
        let mask_weights_id = mask_tape.leaf(mask_weights.clone());
        let masked = mask_tape.causal_mask(mask_input_id).unwrap();
        let mask_loss = mask_tape.mul(masked, mask_weights_id).unwrap();
        let mask_gradients = mask_tape.backward(mask_loss).unwrap();
        check_gradient(&mask_gradients[&mask_input_id], &mask_input, |value| {
            let mut data = value.data().to_vec();
            for row in 0..3 {
                for column in row + 1..3 {
                    data[row * 3 + column] = -1e9;
                }
            }
            Tensor::from_vec(vec![3, 3], data)
                .unwrap()
                .mul(&mask_weights)
                .unwrap()
                .data()
                .iter()
                .sum()
        });
    }

    fn embedding_objective(table: &Tensor, indices: &[usize], weights: &Tensor) -> f32 {
        let columns = table.shape()[1];
        indices
            .iter()
            .enumerate()
            .map(|(row, &index)| {
                (0..columns)
                    .map(|column| {
                        table.data()[index * columns + column]
                            * weights.data()[row * columns + column]
                    })
                    .sum::<f32>()
            })
            .sum()
    }

    #[test]
    fn embedding_backward_scatter_add_matches_finite_differences() {
        let table = Tensor::from_vec(
            vec![4, 3],
            vec![
                0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8, 0.9, 1.0, 1.1, -1.2,
            ],
        )
        .unwrap();
        let indices = vec![1, 2, 1];
        let weights = Tensor::from_vec(
            vec![3, 3],
            vec![0.2, -0.3, 0.4, 0.7, 0.1, -0.5, -0.6, 0.8, 0.9],
        )
        .unwrap();
        let mut tape = Tape::new();
        let table_id = tape.leaf(table.clone());
        let weights_id = tape.leaf(weights.clone());
        let selected = tape.embedding(table_id, indices.clone()).unwrap();
        let loss = tape.mul(selected, weights_id).unwrap();
        let gradients = tape.backward(loss).unwrap();
        assert_eq!(
            &gradients[&table_id].data()[3..6],
            &[
                weights.data()[0] + weights.data()[6],
                weights.data()[1] + weights.data()[7],
                weights.data()[2] + weights.data()[8]
            ]
        );
        check_gradient(&gradients[&table_id], &table, |value| {
            embedding_objective(&value, &indices, &weights)
        });
    }

    fn layer_norm_objective(x: &Tensor, gamma: &Tensor, beta: &Tensor, weights: &Tensor) -> f32 {
        layer_norm(x, gamma, beta, 1e-5)
            .unwrap()
            .0
            .mul(weights)
            .unwrap()
            .data()
            .iter()
            .sum()
    }

    #[test]
    fn layer_norm_tape_backward_matches_finite_differences() {
        let x = Tensor::from_vec(vec![2, 3], vec![0.2, -0.3, 0.7, 1.1, -0.4, 0.9]).unwrap();
        let gamma = Tensor::from_vec(vec![3], vec![1.0, 0.7, 1.2]).unwrap();
        let beta = Tensor::from_vec(vec![3], vec![0.1, 0.2, -0.1]).unwrap();
        let weights = Tensor::from_vec(vec![2, 3], vec![0.3, -0.2, 0.5, -0.1, 0.6, 0.2]).unwrap();
        let mut tape = Tape::new();
        let x_id = tape.leaf(x.clone());
        let gamma_id = tape.leaf(gamma.clone());
        let beta_id = tape.leaf(beta.clone());
        let weights_id = tape.leaf(weights.clone());
        let normalized = tape.layer_norm(x_id, gamma_id, beta_id, 1e-5).unwrap();
        let loss = tape.mul(normalized, weights_id).unwrap();
        let gradients = tape.backward(loss).unwrap();
        check_gradient(&gradients[&x_id], &x, |value| {
            layer_norm_objective(&value, &gamma, &beta, &weights)
        });
        check_gradient(&gradients[&gamma_id], &gamma, |value| {
            layer_norm_objective(&x, &value, &beta, &weights)
        });
        check_gradient(&gradients[&beta_id], &beta, |value| {
            layer_norm_objective(&x, &gamma, &value, &weights)
        });
    }

    #[test]
    fn gelu_and_reshape_backward_match_finite_differences() {
        let input = Tensor::from_vec(vec![2, 3], vec![-1.2, -0.4, 0.0, 0.3, 0.8, 1.5]).unwrap();
        let weights = Tensor::from_vec(vec![3, 2], vec![0.2, -0.5, 0.7, 0.1, -0.3, 0.9]).unwrap();
        let mut tape = Tape::new();
        let input_id = tape.leaf(input.clone());
        let weights_id = tape.leaf(weights.clone());
        let activated = tape.gelu(input_id).unwrap();
        let reshaped = tape.reshape(activated, vec![3, 2]).unwrap();
        let loss = tape.mul(reshaped, weights_id).unwrap();
        let gradients = tape.backward(loss).unwrap();
        check_gradient(&gradients[&input_id], &input, |value| {
            Tensor::from_vec(
                value.shape().to_vec(),
                value.data().iter().copied().map(gelu_value).collect(),
            )
            .unwrap()
            .reshape(vec![3, 2])
            .unwrap()
            .mul(&weights)
            .unwrap()
            .data()
            .iter()
            .sum()
        });
    }

    #[test]
    fn detach_stops_gradient_flow() {
        let mut tape = Tape::new();
        let input = tape.leaf(Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap());
        let tracked = tape.mul(input, input).unwrap();
        let detached = tape.detach(tracked).unwrap();
        let loss = tape.mul(detached, detached).unwrap();
        let gradients = tape.backward(loss).unwrap();
        assert!(!gradients.contains_key(&input));
        assert!(!gradients.contains_key(&tracked));
        assert_eq!(gradients[&detached].data(), &[8.0, 18.0]);
    }

    #[test]
    fn activation_backwards_match_finite_differences() {
        let values = Tensor::from_vec(vec![4], vec![-1.2, -0.4, 0.3, 1.5]).unwrap();
        for activation in 0..3 {
            let mut tape = Tape::new();
            let input = tape.leaf(values.clone());
            let output = match activation {
                0 => tape.relu(input).unwrap(),
                1 => tape.sigmoid(input).unwrap(),
                _ => tape.tanh(input).unwrap(),
            };
            let gradients = tape.backward(output).unwrap();
            check_gradient(&gradients[&input], &values, |value| {
                let output = match activation {
                    0 => value.relu().unwrap(),
                    1 => value.sigmoid().unwrap(),
                    _ => value.tanh().unwrap(),
                };
                output.data().iter().sum()
            });
        }
    }

    fn cross_entropy_value(logits: Tensor, targets: &[usize]) -> f32 {
        let mut tape = Tape::new();
        let logits = tape.leaf(logits);
        let loss = tape.cross_entropy(logits, targets.to_vec()).unwrap();
        tape.value(loss).unwrap().data()[0]
    }

    #[test]
    fn cross_entropy_backward_matches_finite_differences() {
        let logits =
            Tensor::from_vec(vec![2, 4], vec![0.2, -0.4, 0.7, 0.1, 1.1, 0.3, -0.2, 0.8]).unwrap();
        let targets = vec![2, 0];
        let mut tape = Tape::new();
        let logits_id = tape.leaf(logits.clone());
        let loss = tape.cross_entropy(logits_id, targets.clone()).unwrap();
        let gradients = tape.backward(loss).unwrap();
        check_gradient(&gradients[&logits_id], &logits, |value| {
            cross_entropy_value(value, &targets)
        });
    }

    #[test]
    fn uniform_cross_entropy_is_log_vocabulary_size() {
        let logits = Tensor::zeros(vec![3, 65]).unwrap();
        let loss = cross_entropy_value(logits, &[0, 17, 64]);
        assert!((loss - 65.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn reset_preserves_leaves_and_overwrite_rejects_computed_values() {
        let mut tape = Tape::new();
        let left = tape.leaf(Tensor::from_vec(vec![1], vec![1.0]).unwrap());
        let right = tape.leaf(Tensor::from_vec(vec![1], vec![2.0]).unwrap());
        let checkpoint = tape.entry_count();
        let sum = tape.add(left, right).unwrap();
        assert!(matches!(
            tape.overwrite_leaf(sum, Tensor::from_vec(vec![1], vec![4.0]).unwrap()),
            Err(TapeError::NonLeafUpdate(id)) if id == sum
        ));
        tape.overwrite_leaf(left, Tensor::from_vec(vec![1], vec![3.0]).unwrap())
            .unwrap();
        tape.reset(checkpoint).unwrap();
        assert_eq!(tape.entry_count(), checkpoint);
        assert_eq!(tape.value(left).unwrap().data(), &[3.0]);
        assert_eq!(tape.value(right).unwrap().data(), &[2.0]);
        assert!(matches!(tape.value(sum), Err(TapeError::UnknownTensor(id)) if id == sum));
    }
    #[test]
    fn retain_leaves_compacts_dynamic_graph_without_reusing_ids() {
        let mut tape = Tape::new();
        let parameter = tape.leaf(Tensor::from_vec(vec![1], vec![2.0]).unwrap());
        let input = tape.leaf(Tensor::from_vec(vec![1], vec![3.0]).unwrap());
        let output = tape.mul(parameter, input).unwrap();

        tape.retain_leaves(&[parameter]).unwrap();
        assert_eq!(tape.entry_count(), 1);
        assert_eq!(tape.value(parameter).unwrap().data(), &[2.0]);
        assert!(matches!(
            tape.value(input),
            Err(TapeError::UnknownTensor(id)) if id == input
        ));
        assert!(matches!(
            tape.value(output),
            Err(TapeError::UnknownTensor(id)) if id == output
        ));

        let next = tape.leaf(Tensor::from_vec(vec![1], vec![4.0]).unwrap());
        assert!(next > output);
    }

    fn traced_add(
        emitter: Option<EventEmitter>,
        enabled: bool,
    ) -> (Tensor, HashMap<TensorId, Tensor>) {
        let mut tape = Tape::new();
        tape.set_event_emitter(emitter);
        tape.set_trace(11, enabled);
        let left = tape.leaf(Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap());
        let right = tape.leaf(Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap());
        let sum = tape.add(left, right).unwrap();
        (
            tape.value(sum).unwrap().clone(),
            tape.backward(sum).unwrap(),
        )
    }

    #[test]
    fn disabled_tracing_is_byte_identical_and_emits_nothing() {
        let plain = traced_add(None, false);
        let (emitter, receiver) = crate::visualizer::event_channel(16);
        let disabled = traced_add(Some(emitter), false);
        assert_eq!(disabled.0, plain.0);
        assert_eq!(disabled.1, plain.1);
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn tape_emits_forward_then_reverse_backward_with_norms() {
        let (emitter, receiver) = crate::visualizer::event_channel(16);
        traced_add(Some(emitter), true);
        let events: Vec<_> = receiver.try_iter().collect();
        assert_eq!(events.len(), 6);
        assert!(events[0].contains("\"phase\":\"forward\",\"op\":\"leaf\",\"output_id\":0"));
        assert!(events[1].contains("\"phase\":\"forward\",\"op\":\"leaf\",\"output_id\":1"));
        assert!(events[2].contains("\"phase\":\"forward\",\"op\":\"add\",\"output_id\":2"));
        assert!(events[3].contains("\"phase\":\"backward\",\"op\":\"add\",\"output_id\":2"));
        assert!(events[4].contains("\"phase\":\"backward\",\"op\":\"leaf\",\"output_id\":1"));
        assert!(events[5].contains("\"phase\":\"backward\",\"op\":\"leaf\",\"output_id\":0"));
        assert!(events[..3].iter().all(|event| !event.contains("grad_norm")));
        assert!(events[3..].iter().all(|event| event.contains("grad_norm")));
        assert!(events.iter().all(|event| event.contains("\"step\":11")));
    }
}
