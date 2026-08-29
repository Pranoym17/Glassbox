use crate::{Tensor, TensorError};
fn dims(x: &Tensor) -> Result<(usize, usize), TensorError> {
    if x.shape().len() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: x.shape().len(),
        });
    }
    Ok((x.shape()[0], x.shape()[1]))
}
pub fn softmax(x: &Tensor) -> Result<Tensor, TensorError> {
    let (r, c) = dims(x)?;
    let mut o = vec![0.; r * c];
    for i in 0..r {
        let row = &x.data()[i * c..(i + 1) * c];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
        for j in 0..c {
            o[i * c + j] = (row[j] - max).exp() / sum
        }
    }
    Tensor::from_vec(vec![r, c], o)
}
pub fn softmax_backward(s: &Tensor, dy: &Tensor) -> Result<Tensor, TensorError> {
    let (r, c) = dims(s)?;
    if s.shape() != dy.shape() {
        return Err(TensorError::IncompatibleShapes {
            left: s.shape().to_vec(),
            right: dy.shape().to_vec(),
        });
    }
    let mut o = vec![0.; r * c];
    for i in 0..r {
        let dot: f32 = (0..c)
            .map(|j| s.data()[i * c + j] * dy.data()[i * c + j])
            .sum();
        for j in 0..c {
            o[i * c + j] = s.data()[i * c + j] * (dy.data()[i * c + j] - dot)
        }
    }
    Tensor::from_vec(vec![r, c], o)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 5e-3, "{a} {b}")
    }
    #[test]
    fn stable_and_backward_fd() {
        let x = Tensor::from_vec(vec![2, 3], vec![1000.0, 1001.0, 999.0, 0.2, -0.3, 0.7]).unwrap();
        let s = softmax(&x).unwrap();
        for i in 0..2 {
            close(
                s.data()[i * 3] + s.data()[i * 3 + 1] + s.data()[i * 3 + 2],
                1.,
            )
        }
        let dy = Tensor::from_vec(vec![2, 3], vec![0.2, -0.5, 0.7, 1.0, -0.2, 0.3]).unwrap();
        let dx = softmax_backward(&s, &dy).unwrap();
        let e = 1e-3;
        for k in 0..6 {
            let mut p = x.data().to_vec();
            let mut n = p.clone();
            p[k] += e;
            n[k] -= e;
            let fp: f32 = softmax(&Tensor::from_vec(vec![2, 3], p).unwrap())
                .unwrap()
                .data()
                .iter()
                .zip(dy.data())
                .map(|(a, b)| a * b)
                .sum();
            let fm: f32 = softmax(&Tensor::from_vec(vec![2, 3], n).unwrap())
                .unwrap()
                .data()
                .iter()
                .zip(dy.data())
                .map(|(a, b)| a * b)
                .sum();
            close(dx.data()[k], (fp - fm) / (2. * e))
        }
    }
}
