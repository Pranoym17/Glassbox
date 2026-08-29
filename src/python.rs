use crate::Tensor;
use crate::autograd::{Tape, TensorId};
use crate::nn::{
    Embedding as RustEmbedding, Gpt, Initializer, LayerNorm as RustLayerNorm, Linear as RustLinear,
    Module as RustModule,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
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

#[pyclass(name = "GPT", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyGpt {
    model: Gpt,
}

#[pymethods]
impl PyGpt {
    #[new]
    #[pyo3(signature = (vocab_size, block_size, n_layer, n_head, n_embd, seed = 1337))]
    fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
        seed: u64,
    ) -> PyResult<PyClassInitializer<Self>> {
        let model = Gpt::new_with_seed(vocab_size, block_size, n_layer, n_head, n_embd, seed)
            .map_err(value_error)?;
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self { model }))
    }

    fn forward(&mut self, tokens: Vec<Vec<usize>>) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.model.forward(&tokens).map_err(value_error)?,
        })
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

#[pymodule]
pub fn glassbox(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensor>()?;
    let nn = PyModule::new(py, "glassbox.nn")?;
    nn.add_class::<PyModuleBase>()?;
    nn.add_class::<PyLinear>()?;
    nn.add_class::<PyLayerNorm>()?;
    nn.add_class::<PyEmbedding>()?;
    nn.add_class::<PyGpt>()?;
    module.add_submodule(&nn)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
