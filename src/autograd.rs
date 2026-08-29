use crate::{
    Tensor, TensorError,
    transformer::{softmax, softmax_backward},
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
}
#[derive(Clone, Debug)]
pub enum SavedContext {
    None,
    Binary { left: Tensor, right: Tensor },
    Scale { factor: f32 },
    Softmax { output: Tensor },
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
}
