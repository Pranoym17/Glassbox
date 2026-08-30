use crate::Tensor;
use crate::autograd::{Tape, TensorId};
use crate::data::{Batch, CharDataset, Split};
use crate::nn::{
    Embedding as RustEmbedding, Gpt, Initializer, LayerNorm as RustLayerNorm, Linear as RustLinear,
    Module as RustModule,
};
use crate::optim::{Adam, AdamError, Sgd, clip_gradients, gradient_norms};
use crate::visualizer::VisualizerServer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap;
use std::fmt::Display;

fn value_error(error: impl Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "Tensor", module = "glassbox")]
pub struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: Tensor::from_vec(shape, data).map_err(value_error)?,
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn data(&self) -> Vec<f32> {
        self.inner.data().to_vec()
    }

    fn add(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.add(&other.inner).map_err(value_error)?,
        })
    }

    fn mul(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.mul(&other.inner).map_err(value_error)?,
        })
    }

    fn sub(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.sub(&other.inner).map_err(value_error)?,
        })
    }

    fn div(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.div(&other.inner).map_err(value_error)?,
        })
    }

    fn matmul(&self, other: &Self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.matmul(&other.inner).map_err(value_error)?,
        })
    }

    fn relu(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.relu().map_err(value_error)?,
        })
    }

    fn sigmoid(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.sigmoid().map_err(value_error)?,
        })
    }

    fn tanh(&self) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.tanh().map_err(value_error)?,
        })
    }

    fn detach(&self) -> Self {
        Self {
            inner: self.inner.detach(),
        }
    }

    #[pyo3(signature = (other, atol = 1e-8, rtol = 1e-5))]
    fn isclose(&self, other: &Self, atol: f32, rtol: f32) -> PyResult<bool> {
        self.inner
            .is_close(&other.inner, atol, rtol)
            .map_err(value_error)
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.add(other)
    }

    fn __mul__(&self, other: &Self) -> PyResult<Self> {
        self.mul(other)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        self.sub(other)
    }

    fn __truediv__(&self, other: &Self) -> PyResult<Self> {
        self.div(other)
    }

    fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        self.matmul(other)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, data={:?})",
            self.inner.shape(),
            self.inner.data()
        )
    }
}

#[pyclass(name = "Module", subclass, module = "glassbox.nn")]
pub struct PyModuleBase;

#[pyfunction]
#[pyo3(signature = (left, right, atol = 1e-8, rtol = 1e-5))]
fn isclose(left: &PyTensor, right: &PyTensor, atol: f32, rtol: f32) -> PyResult<bool> {
    left.inner
        .is_close(&right.inner, atol, rtol)
        .map_err(value_error)
}

#[pyclass(name = "ReLU", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyReLU;

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyModuleBase).add_subclass(Self)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: input.inner.relu().map_err(value_error)?,
        })
    }
}

#[pyclass(name = "Sigmoid", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PySigmoid;

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyModuleBase).add_subclass(Self)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: input.inner.sigmoid().map_err(value_error)?,
        })
    }
}

#[pyclass(name = "Tanh", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyTanh;

#[pymethods]
impl PyTanh {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyModuleBase).add_subclass(Self)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: input.inner.tanh().map_err(value_error)?,
        })
    }
}

fn parameters(tape: &Tape, ids: Vec<TensorId>) -> PyResult<Vec<PyTensor>> {
    ids.into_iter()
        .map(|id| {
            Ok(PyTensor {
                inner: tape.value(id).map_err(value_error)?.clone(),
            })
        })
        .collect()
}

#[pyclass(name = "Linear", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyLinear {
    tape: Tape,
    layer: RustLinear,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (input_features, output_features, seed = 1337))]
    fn new(
        input_features: usize,
        output_features: usize,
        seed: u64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut tape = Tape::new();
        let mut initializer = Initializer::new(seed);
        let layer = RustLinear::new(&mut tape, input_features, output_features, &mut initializer)
            .map_err(value_error)?;
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self { tape, layer }))
    }

    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        let input = self.tape.leaf(input.inner.clone());
        let output = self
            .layer
            .forward(&mut self.tape, input)
            .map_err(value_error)?;
        Ok(PyTensor {
            inner: self.tape.value(output).map_err(value_error)?.clone(),
        })
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        parameters(&self.tape, self.layer.parameters())
    }
}

#[pyclass(name = "LayerNorm", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyLayerNorm {
    tape: Tape,
    layer: RustLayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (features, epsilon = 1e-5))]
    fn new(features: usize, epsilon: f32) -> PyResult<PyClassInitializer<Self>> {
        let mut tape = Tape::new();
        let layer = RustLayerNorm::new(&mut tape, features, epsilon).map_err(value_error)?;
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self { tape, layer }))
    }

    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        let input = self.tape.leaf(input.inner.clone());
        let output = self
            .layer
            .forward(&mut self.tape, input)
            .map_err(value_error)?;
        Ok(PyTensor {
            inner: self.tape.value(output).map_err(value_error)?.clone(),
        })
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        parameters(&self.tape, self.layer.parameters())
    }
}

#[pyclass(name = "Embedding", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyEmbedding {
    tape: Tape,
    layer: RustEmbedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (entries, features, seed = 1337))]
    fn new(entries: usize, features: usize, seed: u64) -> PyResult<PyClassInitializer<Self>> {
        let mut tape = Tape::new();
        let mut initializer = Initializer::new(seed);
        let layer = RustEmbedding::new(&mut tape, entries, features, &mut initializer)
            .map_err(value_error)?;
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self { tape, layer }))
    }

    fn forward(&mut self, indices: Vec<usize>) -> PyResult<PyTensor> {
        let output = self
            .layer
            .forward(&mut self.tape, indices)
            .map_err(value_error)?;
        Ok(PyTensor {
            inner: self.tape.value(output).map_err(value_error)?.clone(),
        })
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        parameters(&self.tape, self.layer.parameters())
    }
}

enum TrainerOptimizer {
    Adam(Adam),
    Sgd(Sgd),
}

impl TrainerOptimizer {
    fn step(
        &mut self,
        tape: &mut Tape,
        parameters: &[TensorId],
        gradients: &HashMap<TensorId, Tensor>,
    ) -> Result<(), AdamError> {
        match self {
            Self::Adam(optimizer) => optimizer.step(tape, parameters, gradients),
            Self::Sgd(optimizer) => optimizer.step(tape, parameters, gradients),
        }
    }

    fn zero_grad(&self, gradients: &mut HashMap<TensorId, Tensor>) {
        match self {
            Self::Adam(optimizer) => optimizer.zero_grad(gradients),
            Self::Sgd(optimizer) => optimizer.zero_grad(gradients),
        }
    }
}

#[pyclass(name = "GPT", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyGpt {
    model: Gpt,
    optimizer: TrainerOptimizer,
    visualizer: Option<VisualizerServer>,
    step: u64,
    trace_interval: u64,
}

#[pymethods]
impl PyGpt {
    #[new]
    #[pyo3(signature = (vocab_size, block_size, n_layer, n_head, n_embd, seed = 1337, learning_rate = 3e-4, optimizer = "adam", weight_decay = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
        seed: u64,
        learning_rate: f32,
        optimizer: &str,
        weight_decay: f32,
    ) -> PyResult<PyClassInitializer<Self>> {
        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("weight_decay must be non-negative"));
        }
        let optimizer = match optimizer {
            "adam" => TrainerOptimizer::Adam(
                Adam::new(learning_rate, 0.9, 0.999, 1e-8).with_weight_decay(weight_decay),
            ),
            "sgd" if weight_decay == 0.0 => TrainerOptimizer::Sgd(Sgd::new(learning_rate)),
            "sgd" => {
                return Err(PyValueError::new_err(
                    "weight_decay is only supported with optimizer='adam'",
                ));
            }
            value => {
                return Err(PyValueError::new_err(format!(
                    "unknown optimizer {value:?}; use 'adam' or 'sgd'"
                )));
            }
        };
        let model = Gpt::new_with_seed(vocab_size, block_size, n_layer, n_head, n_embd, seed)
            .map_err(value_error)?;
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self {
            model,
            optimizer,
            visualizer: None,
            step: 0,
            trace_interval: 1,
        }))
    }

    fn forward(&mut self, tokens: Vec<Vec<usize>>) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.model.forward(&tokens).map_err(value_error)?,
        })
    }

    #[pyo3(signature = (tokens, targets, maximum_norm = 1.0))]
    fn train_step(
        &mut self,
        tokens: Vec<Vec<usize>>,
        targets: Vec<Vec<usize>>,
        maximum_norm: f32,
    ) -> PyResult<(f32, Vec<f32>)> {
        let traced = self.visualizer.is_some() && self.step.is_multiple_of(self.trace_interval);
        self.model.set_trace(self.step, traced);
        let result: PyResult<(f32, Vec<f32>)> = (|| {
            self.model.reset_tape().map_err(value_error)?;
            let loss = self.model.loss(&tokens, &targets).map_err(value_error)?;
            let value = self.model.loss_value(loss).map_err(value_error)?;
            let mut gradients = self.model.backward(loss).map_err(value_error)?;
            let parameters = self.model.parameters();
            let norms = gradient_norms(&parameters, &gradients);
            clip_gradients(&parameters, &mut gradients, maximum_norm).map_err(value_error)?;
            self.optimizer
                .step(self.model.tape_mut(), &parameters, &gradients)
                .map_err(value_error)?;
            self.optimizer.zero_grad(&mut gradients);
            self.model.reset_tape().map_err(value_error)?;
            Ok((value, norms))
        })();
        self.model.set_trace(self.step, false);
        if result.is_ok() {
            self.step += 1;
        }
        result
    }

    #[pyo3(signature = (port = 8080, trace_interval = 50, capacity = 8192))]
    fn enable_visualizer(
        &mut self,
        port: u16,
        trace_interval: u64,
        capacity: usize,
    ) -> PyResult<String> {
        if trace_interval == 0 {
            return Err(PyValueError::new_err("trace_interval must be positive"));
        }
        self.disable_visualizer();
        let server = VisualizerServer::start(port, capacity).map_err(value_error)?;
        let url = server.url();
        self.model.set_event_emitter(Some(server.emitter()));
        self.trace_interval = trace_interval;
        self.visualizer = Some(server);
        Ok(url)
    }

    fn disable_visualizer(&mut self) {
        self.model.set_event_emitter(None);
        self.model.set_trace(self.step, false);
        self.visualizer = None;
    }

    fn dropped_events(&self) -> u64 {
        self.visualizer
            .as_ref()
            .map(VisualizerServer::dropped)
            .unwrap_or(0)
    }

    fn evaluate(&mut self, tokens: Vec<Vec<usize>>, targets: Vec<Vec<usize>>) -> PyResult<f32> {
        self.model.reset_tape().map_err(value_error)?;
        let loss = self.model.loss(&tokens, &targets).map_err(value_error)?;
        let value = self.model.loss_value(loss).map_err(value_error)?;
        self.model.reset_tape().map_err(value_error)?;
        Ok(value)
    }

    fn save_checkpoint(&self, path: &str) -> PyResult<()> {
        self.model.save_checkpoint(path).map_err(value_error)
    }

    fn load_checkpoint(&mut self, path: &str) -> PyResult<()> {
        self.model.load_checkpoint(path).map_err(value_error)
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(self
            .model
            .parameter_values()
            .map_err(value_error)?
            .into_iter()
            .map(|inner| PyTensor { inner })
            .collect())
    }
}

#[pyclass(name = "CharDataset", module = "glassbox.data")]
pub struct PyCharDataset {
    dataset: CharDataset,
}

#[pymethods]
impl PyCharDataset {
    #[new]
    #[pyo3(signature = (path, block_size, seed = 1337))]
    fn new(path: &str, block_size: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            dataset: CharDataset::from_file(path, block_size, seed).map_err(value_error)?,
        })
    }

    #[getter]
    fn vocabulary(&self) -> Vec<String> {
        self.dataset
            .vocabulary()
            .iter()
            .map(char::to_string)
            .collect()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.dataset.vocab_size()
    }

    fn batch(&mut self, split: &str, batch_size: usize) -> PyResult<Batch> {
        let split = match split {
            "train" => Split::Train,
            "validation" | "val" => Split::Validation,
            value => {
                return Err(PyValueError::new_err(format!(
                    "unknown split {value:?}; use 'train' or 'validation'"
                )));
            }
        };
        self.dataset.batch(split, batch_size).map_err(value_error)
    }
}

#[pymodule]
pub fn glassbox(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensor>()?;
    module.add_function(wrap_pyfunction!(isclose, module)?)?;
    let nn = PyModule::new(py, "glassbox.nn")?;
    nn.add_class::<PyModuleBase>()?;
    nn.add_class::<PyLinear>()?;
    nn.add_class::<PyLayerNorm>()?;
    nn.add_class::<PyEmbedding>()?;
    nn.add_class::<PyReLU>()?;
    nn.add_class::<PySigmoid>()?;
    nn.add_class::<PyTanh>()?;
    nn.add_class::<PyGpt>()?;
    module.add_submodule(&nn)?;
    let data = PyModule::new(py, "glassbox.data")?;
    data.add_class::<PyCharDataset>()?;
    module.add_submodule(&data)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
