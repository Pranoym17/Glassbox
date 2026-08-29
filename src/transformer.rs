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
#[derive(Clone, Debug)]
pub struct LayerNormContext {
    pub normalized: Tensor,
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
}
pub fn layer_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<(Tensor, LayerNormContext), TensorError> {
    let (r, c) = dims(x)?;
    if gamma.shape() != [c] || beta.shape() != [c] {
        return Err(TensorError::IncompatibleShapes {
            left: gamma.shape().to_vec(),
            right: beta.shape().to_vec(),
        });
    }
    let mut y = vec![0.; r * c];
    let mut z = vec![0.; r * c];
    let mut m = vec![];
    let mut v = vec![];
    for i in 0..r {
        let row = &x.data()[i * c..(i + 1) * c];
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / c as f32;
        m.push(mean);
        v.push(var);
        for j in 0..c {
            z[i * c + j] = (row[j] - mean) / (var + eps).sqrt();
            y[i * c + j] = z[i * c + j] * gamma.data()[j] + beta.data()[j]
        }
    }
    let n = Tensor::from_vec(vec![r, c], z)?;
    Ok((
        Tensor::from_vec(vec![r, c], y)?,
        LayerNormContext {
            normalized: n,
            mean: m,
            variance: v,
        },
    ))
}
pub fn layer_norm_backward(
    dy: &Tensor,
    ctx: &LayerNormContext,
    gamma: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor), TensorError> {
    let (r, c) = dims(dy)?;
    let mut dx = vec![0.; r * c];
    let mut dg = vec![0.; c];
    let mut db = vec![0.; c];
    for i in 0..r {
        let inv = (ctx.variance[i] + eps).sqrt().recip();
        let mut a = 0.;
        let mut b = 0.;
        for j in 0..c {
            let q = dy.data()[i * c + j] * gamma.data()[j];
            a += q;
            b += q * ctx.normalized.data()[i * c + j];
            dg[j] += dy.data()[i * c + j] * ctx.normalized.data()[i * c + j];
            db[j] += dy.data()[i * c + j]
        }
        for j in 0..c {
            let q = dy.data()[i * c + j] * gamma.data()[j];
            dx[i * c + j] =
                inv * (q - a / c as f32 - ctx.normalized.data()[i * c + j] * b / c as f32)
        }
    }
    Ok((
        Tensor::from_vec(vec![r, c], dx)?,
        Tensor::from_vec(vec![c], dg)?,
        Tensor::from_vec(vec![c], db)?,
    ))
}
#[cfg(test)]
mod layer_tests {
    use super::*;
    #[test]
    fn layernorm_fd() {
        let x = Tensor::from_vec(vec![2, 3], vec![0.2, -0.3, 0.7, 1.1, -0.4, 0.9]).unwrap();
        let g = Tensor::from_vec(vec![3], vec![1.0, 0.7, 1.2]).unwrap();
        let b = Tensor::from_vec(vec![3], vec![0.1, 0.2, -0.1]).unwrap();
        let dy = Tensor::from_vec(vec![2, 3], vec![0.3, -0.2, 0.5, -0.1, 0.6, 0.2]).unwrap();
        let (_, c) = layer_norm(&x, &g, &b, 1e-5).unwrap();
        let (dx, _, _) = layer_norm_backward(&dy, &c, &g, 1e-5).unwrap();
        let e = 1e-3;
        for k in 0..6 {
            let mut p = x.data().to_vec();
            let mut n = p.clone();
            p[k] += e;
            n[k] -= e;
            let f = |z: Vec<f32>| -> f32 {
                layer_norm(&Tensor::from_vec(vec![2, 3], z).unwrap(), &g, &b, 1e-5)
                    .unwrap()
                    .0
                    .data()
                    .iter()
                    .zip(dy.data())
                    .map(|(a, b)| a * b)
                    .sum()
            };
            assert!((dx.data()[k] - (f(p) - f(n)) / (2. * e)).abs() < 5e-3)
        }
    }
}
