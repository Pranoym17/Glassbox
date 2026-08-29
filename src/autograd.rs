use crate::{Tensor, TensorError};
use std::collections::{HashMap, HashSet};
use std::fmt;
pub type TensorId = usize;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpType {
    Leaf,
    Add,
    Mul,
}
#[derive(Clone, Debug)]
pub enum SavedContext {
    None,
    Binary { left: Tensor, right: Tensor },
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
}
