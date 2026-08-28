use std::fmt;

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
    IncompatibleShapes {
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
            } => {
                write!(
                    f,
                    "index {index} is out of bounds for axis {axis} with size {dimension}"
                )
            }
            Self::InvalidPermutation => write!(f, "axes must be a permutation of tensor axes"),
            Self::IncompatibleShapes { left, right } => {
                write!(f, "cannot broadcast shapes {left:?} and {right:?}")
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

    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, |left, right| left + right)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, |left, right| left * right)
    }

    fn binary_op(
        &self,
        rhs: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let output_shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let output_strides = contiguous_strides(&output_shape);
        let mut output = Vec::with_capacity(numel(&output_shape));

        for flat_index in 0..numel(&output_shape) {
            let coordinates = unravel(flat_index, &output_shape, &output_strides);
            let left_offset = broadcast_offset(&coordinates, &output_shape, self);
            let right_offset = broadcast_offset(&coordinates, &output_shape, rhs);
            output.push(operation(self.data[left_offset], rhs.data[right_offset]));
        }
        Self::from_vec(output_shape, output)
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
    fn add_broadcasts_a_vector_over_2d_rows() {
        let matrix = Tensor::from_vec(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let vector = Tensor::from_vec(vec![3], vec![10., 20., 30.]).unwrap();

        let result = matrix.add(&vector).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.data(), &[11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn mul_broadcasts_two_2d_tensors() {
        let column = Tensor::from_vec(vec![2, 1], vec![2., 3.]).unwrap();
        let row = Tensor::from_vec(vec![1, 3], vec![10., 20., 30.]).unwrap();

        let result = column.mul(&row).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.data(), &[20., 40., 60., 30., 60., 90.]);
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
    fn incompatible_shapes_fail_clearly() {
        let left = Tensor::zeros(vec![2, 3]).unwrap();
        let right = Tensor::zeros(vec![2, 2]).unwrap();

        assert_eq!(
            left.add(&right).unwrap_err(),
            TensorError::IncompatibleShapes {
                left: vec![2, 3],
                right: vec![2, 2],
            }
        );
    }
}
