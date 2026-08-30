use crate::Tensor;
use crate::autograd::{Tape, TapeError, TensorId};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug)]
pub enum AdamError {
    Tape(TapeError),
    MissingGradient(TensorId),
}

impl fmt::Display for AdamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tape(error) => write!(f, "{error}"),
            Self::MissingGradient(id) => write!(f, "missing gradient for parameter {id}"),
        }
    }
}

impl std::error::Error for AdamError {}

impl From<TapeError> for AdamError {
    fn from(error: TapeError) -> Self {
        Self::Tape(error)
    }
}
impl From<crate::TensorError> for AdamError {
    fn from(error: crate::TensorError) -> Self {
        Self::Tape(TapeError::Tensor(error))
    }
}

pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: u64,
    first_moment: HashMap<TensorId, Vec<f32>>,
    second_moment: HashMap<TensorId, Vec<f32>>,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            step: 0,
            first_moment: HashMap::new(),
            second_moment: HashMap::new(),
        }
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn step(
        &mut self,
        tape: &mut Tape,
        parameters: &[TensorId],
        gradients: &HashMap<TensorId, Tensor>,
    ) -> Result<(), AdamError> {
        for &id in parameters {
            let gradient = gradients.get(&id).ok_or(AdamError::MissingGradient(id))?;
            if tape.value(id)?.shape() != gradient.shape() {
                return Err(TapeError::Tensor(crate::TensorError::IncompatibleShapes {
                    left: tape.value(id)?.shape().to_vec(),
                    right: gradient.shape().to_vec(),
                })
                .into());
            }
        }

        self.step += 1;
        let first_correction = 1.0 - self.beta1.powf(self.step as f32);
        let second_correction = 1.0 - self.beta2.powf(self.step as f32);
        for &id in parameters {
            let parameter = tape.value(id)?.clone();
            let gradient = &gradients[&id];
            let first = self
                .first_moment
                .entry(id)
                .or_insert_with(|| vec![0.0; parameter.data().len()]);
            let second = self
                .second_moment
                .entry(id)
                .or_insert_with(|| vec![0.0; parameter.data().len()]);
            let mut values = parameter.data().to_vec();
            for index in 0..values.len() {
                first[index] =
                    self.beta1 * first[index] + (1.0 - self.beta1) * gradient.data()[index];
                second[index] = self.beta2 * second[index]
                    + (1.0 - self.beta2) * gradient.data()[index].powi(2);
                let corrected_first = first[index] / first_correction;
                let corrected_second = second[index] / second_correction;
                let adaptive = corrected_first / (corrected_second.sqrt() + self.epsilon);
                values[index] -=
                    self.learning_rate * (adaptive + self.weight_decay * parameter.data()[index]);
            }
            tape.overwrite_leaf(id, Tensor::from_vec(parameter.shape().to_vec(), values)?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&self, gradients: &mut HashMap<TensorId, Tensor>) {
        gradients.clear();
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new(3e-4, 0.9, 0.999, 1e-8)
    }
}

pub struct Sgd {
    learning_rate: f32,
}

impl Sgd {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    pub fn step(
        &mut self,
        tape: &mut Tape,
        parameters: &[TensorId],
        gradients: &HashMap<TensorId, Tensor>,
    ) -> Result<(), AdamError> {
        for &id in parameters {
            let parameter = tape.value(id)?;
            let gradient = gradients.get(&id).ok_or(AdamError::MissingGradient(id))?;
            if parameter.shape() != gradient.shape() {
                return Err(TapeError::Tensor(crate::TensorError::IncompatibleShapes {
                    left: parameter.shape().to_vec(),
                    right: gradient.shape().to_vec(),
                })
                .into());
            }
        }
        for &id in parameters {
            let parameter = tape.value(id)?.clone();
            let gradient = &gradients[&id];
            let values = parameter
                .data()
                .iter()
                .zip(gradient.data())
                .map(|(value, gradient)| value - self.learning_rate * gradient)
                .collect();
            tape.overwrite_leaf(id, Tensor::from_vec(parameter.shape().to_vec(), values)?)?;
        }
        Ok(())
    }

    pub fn zero_grad(&self, gradients: &mut HashMap<TensorId, Tensor>) {
        gradients.clear();
    }
}

pub fn gradient_norms(parameters: &[TensorId], gradients: &HashMap<TensorId, Tensor>) -> Vec<f32> {
    parameters
        .iter()
        .map(|id| {
            gradients
                .get(id)
                .map(|gradient| {
                    gradient
                        .data()
                        .iter()
                        .map(|value| value * value)
                        .sum::<f32>()
                        .sqrt()
                })
                .unwrap_or(0.0)
        })
        .collect()
}

pub fn clip_gradients(
    parameters: &[TensorId],
    gradients: &mut HashMap<TensorId, Tensor>,
    maximum_norm: f32,
) -> Result<f32, TapeError> {
    let norm = parameters
        .iter()
        .filter_map(|id| gradients.get(id))
        .flat_map(|gradient| gradient.data())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > maximum_norm {
        let scale = maximum_norm / (norm + 1e-12);
        for id in parameters {
            if let Some(gradient) = gradients.get_mut(id) {
                *gradient = Tensor::from_vec(
                    gradient.shape().to_vec(),
                    gradient.data().iter().map(|value| value * scale).collect(),
                )?;
            }
        }
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_converges_on_quadratic_without_growing_tape() {
        let mut tape = Tape::new();
        let parameter = tape.leaf(Tensor::from_vec(vec![], vec![0.0]).unwrap());
        let offset = tape.leaf(Tensor::from_vec(vec![], vec![-3.0]).unwrap());
        let checkpoint = tape.entry_count();
        let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);

        for _ in 0..200 {
            let difference = tape.add(parameter, offset).unwrap();
            let loss = tape.mul(difference, difference).unwrap();
            let mut gradients = tape.backward(loss).unwrap();
            let entries = tape.entry_count();
            optimizer.step(&mut tape, &[parameter], &gradients).unwrap();
            assert_eq!(tape.entry_count(), entries);
            optimizer.zero_grad(&mut gradients);
            assert!(gradients.is_empty());
            tape.reset(checkpoint).unwrap();
        }

        assert!((tape.value(parameter).unwrap().data()[0] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn gradient_clipping_uses_global_norm() {
        let first = Tensor::from_vec(vec![1], vec![3.0]).unwrap();
        let second = Tensor::from_vec(vec![1], vec![4.0]).unwrap();
        let mut gradients = HashMap::from([(0, first), (1, second)]);
        let norm = clip_gradients(&[0, 1], &mut gradients, 1.0).unwrap();
        assert!((norm - 5.0).abs() < 1e-6);
        assert!((gradients[&0].data()[0] - 0.6).abs() < 1e-6);
        assert!((gradients[&1].data()[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sgd_converges_on_quadratic() {
        let mut tape = Tape::new();
        let parameter = tape.leaf(Tensor::from_vec(vec![], vec![0.0]).unwrap());
        let offset = tape.leaf(Tensor::from_vec(vec![], vec![-3.0]).unwrap());
        let checkpoint = tape.entry_count();
        let mut optimizer = Sgd::new(0.1);
        for _ in 0..100 {
            let difference = tape.add(parameter, offset).unwrap();
            let loss = tape.mul(difference, difference).unwrap();
            let mut gradients = tape.backward(loss).unwrap();
            optimizer.step(&mut tape, &[parameter], &gradients).unwrap();
            optimizer.zero_grad(&mut gradients);
            tape.reset(checkpoint).unwrap();
        }
        assert!((tape.value(parameter).unwrap().data()[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn adam_weight_decay_is_decoupled_from_gradient() {
        let mut tape = Tape::new();
        let parameter = tape.leaf(Tensor::from_vec(vec![1], vec![2.0]).unwrap());
        let gradients = HashMap::from([(parameter, Tensor::from_vec(vec![1], vec![0.0]).unwrap())]);
        let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8).with_weight_decay(0.2);
        optimizer.step(&mut tape, &[parameter], &gradients).unwrap();
        assert!((tape.value(parameter).unwrap().data()[0] - 1.96).abs() < 1e-6);
    }
}
