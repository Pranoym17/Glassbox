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
    MatMul,
    Transpose,
    Scale,
    CausalMask,
    Softmax,
    Embedding,
    LayerNorm,
    Gelu,
    Reshape,
}
#[derive(Clone, Debug)]
pub enum SavedContext {
    None,
    Binary {
        left: Tensor,
        right: Tensor,
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
    Reshape {
        input_shape: Vec<usize>,
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
}
impl fmt::Display for TapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTensor(id) => write!(f, "unknown tensor {id}"),
            Self::Tensor(e) => write!(f, "tensor error: {e}"),
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
}
impl Tape {
    pub fn new() -> Self {
        Self {
            entries: vec![],
            index: HashMap::new(),
            values: HashMap::new(),
            next_id: 0,
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
                (OpType::Reshape, SavedContext::Reshape { input_shape }) => {
                    self.acc(&mut g, e.inputs[0], up.reshape(input_shape.clone())?)?
                }
                _ => unreachable!(),
            }
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
        id
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
}
