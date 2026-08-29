use std::fmt;

pub mod autograd;
pub mod gpu;
pub mod nn;
mod python;
pub mod transformer;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    ZeroSizedDimension,
    DataLengthMismatch {
        expected: usize,
        actual: usize,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        dimension: usize,
    },
    InvalidPermutation,
    InvalidAxis {
        axis: usize,
        rank: usize,
    },
    IncompatibleShapes {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    IncompatibleMatMul {
        left: Vec<usize>,
        right: Vec<usize>,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSizedDimension => write!(f, "tensor dimensions must be non-zero"),
            Self::DataLengthMismatch { expected, actual } => {
                write!(f, "shape requires {expected} elements, received {actual}")
            }
            Self::RankMismatch { expected, actual } => {
                write!(f, "expected {expected} indices, received {actual}")
            }
            Self::IndexOutOfBounds {
                axis,
                index,
                dimension,
            } => write!(
                f,
                "index {index} is out of bounds for axis {axis} with size {dimension}"
            ),
            Self::InvalidPermutation => write!(f, "axes must be a permutation of tensor axes"),
            Self::InvalidAxis { axis, rank } => {
                write!(f, "axis {axis} is invalid for a tensor with rank {rank}")
            }
            Self::IncompatibleShapes { left, right } => {
                write!(f, "cannot broadcast shapes {left:?} and {right:?}")
            }
            Self::IncompatibleMatMul { left, right } => {
                write!(
                    f,
                    "cannot multiply matrices with shapes {left:?} and {right:?}"
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}

impl Tensor {
    pub fn from_vec(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        if shape.contains(&0) {
            return Err(TensorError::ZeroSizedDimension);
        }
        let expected = numel(&shape);
        if data.len() != expected {
            return Err(TensorError::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            strides: contiguous_strides(&shape),
            shape,
            data,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = numel(&shape);
        Self::from_vec(shape, vec![0.0; count])
    }
    pub fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        let count = numel(&shape);
        Self::from_vec(shape, vec![1.0; count])
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    pub fn get(&self, indices: &[usize]) -> Result<f32, TensorError> {
        Ok(self.data[self.offset(indices)?])
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.shape.len() {
            return Err(TensorError::InvalidPermutation);
        }
        let mut used = vec![false; axes.len()];
        for &axis in axes {
            if axis >= axes.len() || used[axis] {
                return Err(TensorError::InvalidPermutation);
            }
            used[axis] = true;
        }
        Ok(Self {
            shape: axes.iter().map(|&axis| self.shape[axis]).collect(),
            strides: axes.iter().map(|&axis| self.strides[axis]).collect(),
            data: self.data.clone(),
        })
    }

    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        let output_strides = contiguous_strides(&self.shape);
        let data = (0..numel(&self.shape))
            .map(|flat_index| {
                let coordinates = unravel(flat_index, &self.shape, &output_strides);
                self.data[broadcast_offset(&coordinates, &self.shape, self)]
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            strides: output_strides,
            data,
        }
    }
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, TensorError> {
        Self::from_vec(shape, self.contiguous().data)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, |a, b| a + b)
    }
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, |a, b| a * b)
    }
    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, |a, b| a - b)
    }
    pub fn div(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, |a, b| a / b)
    }
    pub fn exp(&self) -> Result<Self, TensorError> {
        self.unary_op(f32::exp)
    }
    pub fn log(&self) -> Result<Self, TensorError> {
        self.unary_op(f32::ln)
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 || self.shape[1] != rhs.shape[0] {
            return Err(TensorError::IncompatibleMatMul {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }
        let left = self.contiguous();
        let right = rhs.contiguous();
        let (rows, inner, columns) = (left.shape[0], left.shape[1], right.shape[1]);
        let mut output = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                for index in 0..inner {
                    output[row * columns + column] +=
                        left.data[row * inner + index] * right.data[index * columns + column];
                }
            }
        }
        Self::from_vec(vec![rows, columns], output)
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.shape.len() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.shape.len(),
            });
        }
        let input = self.contiguous();
        let outer = numel(&input.shape[..axis]);
        let reduce = input.shape[axis];
        let inner = numel(&input.shape[axis + 1..]);
        let mut output = vec![0.0; outer * inner];
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let base = outer_index * reduce * inner + inner_index;
                output[outer_index * inner + inner_index] = (0..reduce)
                    .map(|reduce_index| input.data[base + reduce_index * inner])
                    .sum();
            }
        }
        let mut output_shape = input.shape;
        output_shape.remove(axis);
        Self::from_vec(output_shape, output)
    }

    pub(crate) fn sum_to_shape(&self, target: &[usize]) -> Result<Self, TensorError> {
        if target.len() > self.shape.len() {
            return Err(TensorError::IncompatibleShapes {
                left: self.shape.clone(),
                right: target.to_vec(),
            });
        };
        let d = self.shape.len() - target.len();
        for (i, &v) in target.iter().enumerate() {
            if v != 1 && v != self.shape[i + d] {
                return Err(TensorError::IncompatibleShapes {
                    left: self.shape.clone(),
                    right: target.to_vec(),
                });
            }
        }
        let src = self.contiguous();
        let ss = contiguous_strides(src.shape());
        let ts = contiguous_strides(target);
        let mut out = vec![0.; numel(target)];
        for f in 0..src.data.len() {
            let c = unravel(f, src.shape(), &ss);
            let mut o = 0;
            for (i, (&v, &st)) in target.iter().zip(&ts).enumerate() {
                o += if v == 1 { 0 } else { c[i + d] } * st
            }
            out[o] += src.data[f]
        }
        Self::from_vec(target.to_vec(), out)
    }

    pub(crate) fn broadcast_shape(
        left: &[usize],
        right: &[usize],
    ) -> Result<Vec<usize>, TensorError> {
        broadcast_shape(left, right)
    }

    pub(crate) fn broadcast_to(&self, shape: &[usize]) -> Result<Self, TensorError> {
        if broadcast_shape(&self.shape, shape)? != shape {
            return Err(TensorError::IncompatibleShapes {
                left: self.shape.clone(),
                right: shape.to_vec(),
            });
        }
        let output_strides = contiguous_strides(shape);
        let data = (0..numel(shape))
            .map(|flat_index| {
                let coordinates = unravel(flat_index, shape, &output_strides);
                self.data[broadcast_offset(&coordinates, shape, self)]
            })
            .collect();
        Self::from_vec(shape.to_vec(), data)
    }

    fn unary_op(&self, operation: impl Fn(f32) -> f32) -> Result<Self, TensorError> {
        let input = self.contiguous();
        Self::from_vec(input.shape, input.data.into_iter().map(operation).collect())
    }

    fn binary_op(
        &self,
        rhs: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let output_shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let left = self.broadcast_to(&output_shape)?;
        let right = rhs.broadcast_to(&output_shape)?;
        Self::from_vec(
            output_shape,
            left.data
                .into_iter()
                .zip(right.data)
                .map(|(a, b)| operation(a, b))
                .collect(),
        )
    }

    fn offset(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::RankMismatch {
                expected: self.shape.len(),
                actual: indices.len(),
            });
        }
        let mut offset = 0;
        for (axis, (&index, (&dimension, &stride))) in indices
            .iter()
            .zip(self.shape.iter().zip(&self.strides))
            .enumerate()
        {
            if index >= dimension {
                return Err(TensorError::IndexOutOfBounds {
                    axis,
                    index,
                    dimension,
                });
            }
            offset += index * stride;
        }
        Ok(offset)
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride *= shape[axis];
    }
    strides
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, TensorError> {
    let rank = left.len().max(right.len());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let left_dim = left.iter().rev().nth(offset).copied().unwrap_or(1);
        let right_dim = right.iter().rev().nth(offset).copied().unwrap_or(1);
        if left_dim != right_dim && left_dim != 1 && right_dim != 1 {
            return Err(TensorError::IncompatibleShapes {
                left: left.to_vec(),
                right: right.to_vec(),
            });
        }
        shape.push(left_dim.max(right_dim));
    }
    shape.reverse();
    Ok(shape)
}

fn unravel(mut flat_index: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(strides)
        .map(|(_, &stride)| {
            let coordinate = flat_index / stride;
            flat_index %= stride;
            coordinate
        })
        .collect()
}

fn broadcast_offset(coordinates: &[usize], output_shape: &[usize], input: &Tensor) -> usize {
    let rank_difference = output_shape.len() - input.shape.len();
    input
        .shape
        .iter()
        .zip(&input.strides)
        .enumerate()
        .map(|(axis, (&dimension, &stride))| {
            let coordinate = coordinates[axis + rank_difference];
            if dimension == 1 {
                0
            } else {
                coordinate * stride
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{Tensor, TensorError};

    #[test]
    fn row_major_strides_and_indexing_are_correct() {
        let tensor = Tensor::from_vec(vec![2, 3], vec![0., 1., 2., 3., 4., 5.]).unwrap();
        assert_eq!(tensor.strides(), &[3, 1]);
        assert!(tensor.is_contiguous());
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 5.0);
    }

    #[test]
    fn elementwise_ops_broadcast_and_match_expected_values() {
        let matrix = Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let vector = Tensor::from_vec(vec![3], vec![10., 20., 30.]).unwrap();
        assert_eq!(
            matrix.add(&vector).unwrap().data(),
            &[11., 22., 33., 14., 25., 36.]
        );
        assert_eq!(
            matrix.sub(&vector).unwrap().data(),
            &[-9., -18., -27., -6., -15., -24.]
        );
        assert_eq!(
            matrix.mul(&vector).unwrap().data(),
            &[10., 40., 90., 40., 100., 180.]
        );
        assert_eq!(
            matrix.div(&vector).unwrap().data(),
            &[0.1, 0.1, 0.1, 0.4, 0.25, 0.2]
        );
    }

    #[test]
    fn unary_ops_and_reduction_are_correct() {
        let values = Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(values.sum_axis(0).unwrap().data(), &[5., 7., 9.]);
        assert_eq!(values.sum_axis(1).unwrap().data(), &[6., 15.]);
        for (&actual, &expected) in values
            .exp()
            .unwrap()
            .log()
            .unwrap()
            .data()
            .iter()
            .zip(values.data())
        {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn non_contiguous_views_work_in_elementwise_ops() {
        let matrix = Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let transposed = matrix.permute(&[1, 0]).unwrap();
        let scalar = Tensor::from_vec(vec![], vec![10.]).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
        assert_eq!(transposed.get(&[2, 1]).unwrap(), 6.0);
        assert_eq!(
            transposed.add(&scalar).unwrap().data(),
            &[11., 14., 12., 15., 13., 16.]
        );
    }

    #[test]
    fn invalid_tensor_operations_fail_clearly() {
        let left = Tensor::zeros(vec![2, 3]).unwrap();
        let right = Tensor::zeros(vec![2, 2]).unwrap();
        assert_eq!(
            left.add(&right).unwrap_err(),
            TensorError::IncompatibleShapes {
                left: vec![2, 3],
                right: vec![2, 2]
            }
        );
        assert_eq!(
            left.sum_axis(2).unwrap_err(),
            TensorError::InvalidAxis { axis: 2, rank: 2 }
        );
    }
}
