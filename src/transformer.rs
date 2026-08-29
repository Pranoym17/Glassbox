use crate::autograd::{Tape, TapeError, TensorId};
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
        let (dx, dg, db) = layer_norm_backward(&dy, &c, &g, 1e-5).unwrap();
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
        for k in 0..3 {
            let mut plus = g.data().to_vec();
            let mut minus = plus.clone();
            plus[k] += e;
            minus[k] -= e;
            let evaluate = |values: Vec<f32>| -> f32 {
                layer_norm(&x, &Tensor::from_vec(vec![3], values).unwrap(), &b, 1e-5)
                    .unwrap()
                    .0
                    .data()
                    .iter()
                    .zip(dy.data())
                    .map(|(a, b)| a * b)
                    .sum()
            };
            let numerical = (evaluate(plus) - evaluate(minus)) / (2.0 * e);
            assert!((dg.data()[k] - numerical).abs() < 5e-3);
        }
        for k in 0..3 {
            let mut plus = b.data().to_vec();
            let mut minus = plus.clone();
            plus[k] += e;
            minus[k] -= e;
            let evaluate = |values: Vec<f32>| -> f32 {
                layer_norm(&x, &g, &Tensor::from_vec(vec![3], values).unwrap(), 1e-5)
                    .unwrap()
                    .0
                    .data()
                    .iter()
                    .zip(dy.data())
                    .map(|(a, b)| a * b)
                    .sum()
            };
            let numerical = (evaluate(plus) - evaluate(minus)) / (2.0 * e);
            assert!((db.data()[k] - numerical).abs() < 5e-3);
        }
    }
}
pub fn causal_attention_tape(
    tape: &mut Tape,
    q: TensorId,
    k: TensorId,
    v: TensorId,
) -> Result<TensorId, TapeError> {
    let q_shape = tape.value(q)?.shape().to_vec();
    let k_shape = tape.value(k)?.shape().to_vec();
    let v_shape = tape.value(v)?.shape().to_vec();
    if q_shape.len() != 2 || k_shape != q_shape || v_shape != q_shape {
        return Err(TensorError::IncompatibleShapes {
            left: q_shape,
            right: k_shape,
        }
        .into());
    }
    let dimension = tape.value(q)?.shape()[1];
    let kt = tape.transpose(k)?;
    let scores = tape.matmul(q, kt)?;
    let scaled = tape.scale(scores, 1.0 / (dimension as f32).sqrt())?;
    let masked = tape.causal_mask(scaled)?;
    let weights = tape.softmax(masked)?;
    tape.matmul(weights, v)
}

pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, TensorError> {
    let mut tape = Tape::new();
    let q = tape.leaf(q.clone());
    let k = tape.leaf(k.clone());
    let v = tape.leaf(v.clone());
    let output = causal_attention_tape(&mut tape, q, k, v).map_err(tape_tensor_error)?;
    Ok(tape.value(output).map_err(tape_tensor_error)?.clone())
}

fn tape_tensor_error(error: TapeError) -> TensorError {
    match error {
        TapeError::Tensor(error) => error,
        TapeError::UnknownTensor(id) => panic!("attention produced unknown tensor {id}"),
    }
}
#[cfg(test)]
mod attention_tests {
    use super::*;
    #[test]
    fn causal_attention_matches_manual_reference() {
        let q = Tensor::from_vec(vec![2, 2], vec![1., 0., 0., 1.]).unwrap();
        let k = q.clone();
        let v = Tensor::from_vec(vec![2, 2], vec![2., 3., 5., 7.]).unwrap();
        let o = causal_attention(&q, &k, &v).unwrap();
        assert!((o.data()[0] - 2.).abs() < 1e-5);
        assert!((o.data()[1] - 3.).abs() < 1e-5);
        let w = 1f32 / (1f32 + (-1f32 / 2f32.sqrt()).exp());
        assert!((o.data()[2] - (w * 5. + (1. - w) * 2.)).abs() < 1e-5);
        assert!((o.data()[3] - (w * 7. + (1. - w) * 3.)).abs() < 1e-5)
    }

    fn objective(q: &Tensor, k: &Tensor, v: &Tensor) -> f32 {
        causal_attention(q, k, v).unwrap().data().iter().sum()
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
    fn causal_attention_backward_matches_finite_differences() {
        let values = |offset: usize| {
            Tensor::from_vec(
                vec![4, 8],
                (0..32)
                    .map(|index| ((index * 7 + offset) % 19) as f32 * 0.04 - 0.35)
                    .collect(),
            )
            .unwrap()
        };
        let q_value = values(1);
        let k_value = values(5);
        let v_value = values(9);
        let mut tape = Tape::new();
        let q = tape.leaf(q_value.clone());
        let k = tape.leaf(k_value.clone());
        let v = tape.leaf(v_value.clone());
        let output = causal_attention_tape(&mut tape, q, k, v).unwrap();
        let gradients = tape.backward(output).unwrap();

        check_gradient(&gradients[&q], &q_value, |value| {
            objective(&value, &k_value, &v_value)
        });
        check_gradient(&gradients[&k], &k_value, |value| {
            objective(&q_value, &value, &v_value)
        });
        check_gradient(&gradients[&v], &v_value, |value| {
            objective(&q_value, &k_value, &value)
        });
    }
}
