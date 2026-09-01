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
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;

fn value_error(error: impl Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

struct PythonGraph {
    tape: RefCell<Tape>,
    gradients: RefCell<HashMap<TensorId, Tensor>>,
    parameters: RefCell<Vec<TensorId>>,
}

impl PythonGraph {
    fn new() -> Self {
        Self {
            tape: RefCell::new(Tape::new()),
            gradients: RefCell::new(HashMap::new()),
            parameters: RefCell::new(Vec::new()),
        }
    }

    fn register_parameters(&self, ids: &[TensorId]) {
        let mut parameters = self.parameters.borrow_mut();
        for &id in ids {
            if !parameters.contains(&id) {
                parameters.push(id);
            }
        }
    }

    fn reset(&self) -> PyResult<()> {
        self.tape
            .borrow_mut()
            .retain_leaves(&self.parameters.borrow())
            .map_err(value_error)
    }
}

thread_local! {
    static DEFAULT_GRAPH: Rc<PythonGraph> = Rc::new(PythonGraph::new());
}

fn default_graph() -> Rc<PythonGraph> {
    DEFAULT_GRAPH.with(Rc::clone)
}

#[pyclass(unsendable, name = "Tensor", module = "glassbox")]
pub struct PyTensor {
    inner: Tensor,
    graph: Option<Rc<PythonGraph>>,
    id: Option<TensorId>,
}

impl PyTensor {
    fn eager(inner: Tensor) -> Self {
        Self {
            inner,
            graph: None,
            id: None,
        }
    }

    fn from_graph(graph: Rc<PythonGraph>, id: TensorId) -> PyResult<Self> {
        let inner = graph.tape.borrow().value(id).map_err(value_error)?.clone();
        Ok(Self {
            inner,
            graph: Some(graph),
            id: Some(id),
        })
    }

    fn value(&self) -> PyResult<Tensor> {
        match (&self.graph, self.id) {
            (Some(graph), Some(id)) => graph.tape.borrow().value(id).cloned().map_err(value_error),
            _ => Ok(self.inner.clone()),
        }
    }

    fn id_on(&self, graph: &Rc<PythonGraph>) -> PyResult<TensorId> {
        match (&self.graph, self.id) {
            (Some(current), Some(id)) if Rc::ptr_eq(current, graph) => Ok(id),
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "cannot combine tensors from different autograd tapes",
            )),
            _ => Ok(graph.tape.borrow_mut().leaf(self.inner.clone())),
        }
    }

    fn binary_op(&self, other: &Self, operation: &str) -> PyResult<Self> {
        if let (Some(left), Some(right)) = (&self.graph, &other.graph)
            && !Rc::ptr_eq(left, right)
        {
            return Err(PyValueError::new_err(
                "cannot combine tensors from different autograd tapes",
            ));
        }
        let graph = self.graph.clone().or_else(|| other.graph.clone());
        if let Some(graph) = graph {
            let left = self.id_on(&graph)?;
            let right = other.id_on(&graph)?;
            let output = {
                let mut tape = graph.tape.borrow_mut();
                match operation {
                    "add" => tape.add(left, right),
                    "mul" => tape.mul(left, right),
                    "sub" => tape.sub(left, right),
                    "div" => tape.div(left, right),
                    "matmul" => tape.matmul(left, right),
                    _ => unreachable!(),
                }
                .map_err(value_error)?
            };
            Self::from_graph(graph, output)
        } else {
            let output = match operation {
                "add" => self.inner.add(&other.inner),
                "mul" => self.inner.mul(&other.inner),
                "sub" => self.inner.sub(&other.inner),
                "div" => self.inner.div(&other.inner),
                "matmul" => self.inner.matmul(&other.inner),
                _ => unreachable!(),
            }
            .map_err(value_error)?;
            Ok(Self::eager(output))
        }
    }

    fn unary_op(&self, operation: &str) -> PyResult<Self> {
        if let Some(graph) = self.graph.clone() {
            let input = self.id.expect("graph tensors have ids");
            let output = {
                let mut tape = graph.tape.borrow_mut();
                match operation {
                    "relu" => tape.relu(input),
                    "sigmoid" => tape.sigmoid(input),
                    "tanh" => tape.tanh(input),
                    _ => unreachable!(),
                }
                .map_err(value_error)?
            };
            Self::from_graph(graph, output)
        } else {
            let output = match operation {
                "relu" => self.inner.relu(),
                "sigmoid" => self.inner.sigmoid(),
                "tanh" => self.inner.tanh(),
                _ => unreachable!(),
            }
            .map_err(value_error)?;
            Ok(Self::eager(output))
        }
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self::eager(
            Tensor::from_vec(shape, data).map_err(value_error)?,
        ))
    }

    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.value()?.shape().to_vec())
    }

    #[getter]
    fn data(&self) -> PyResult<Vec<f32>> {
        Ok(self.value()?.data().to_vec())
    }

    fn add(&self, other: &Self) -> PyResult<Self> {
        self.binary_op(other, "add")
    }

    fn mul(&self, other: &Self) -> PyResult<Self> {
        self.binary_op(other, "mul")
    }

    fn sub(&self, other: &Self) -> PyResult<Self> {
        self.binary_op(other, "sub")
    }

    fn div(&self, other: &Self) -> PyResult<Self> {
        self.binary_op(other, "div")
    }

    fn matmul(&self, other: &Self) -> PyResult<Self> {
        self.binary_op(other, "matmul")
    }

    fn relu(&self) -> PyResult<Self> {
        self.unary_op("relu")
    }

    fn sigmoid(&self) -> PyResult<Self> {
        self.unary_op("sigmoid")
    }

    fn tanh(&self) -> PyResult<Self> {
        self.unary_op("tanh")
    }

    /// Returns a shared-storage eager tensor that is disconnected from its source tape.
    fn detach(&self) -> PyResult<Self> {
        Ok(Self::eager(self.value()?.detach()))
    }

    #[pyo3(signature = (other, atol = 1e-8, rtol = 1e-5))]
    fn isclose(&self, other: &Self, atol: f32, rtol: f32) -> PyResult<bool> {
        self.value()?
            .is_close(&other.value()?, atol, rtol)
            .map_err(value_error)
    }

    fn backward(&self) -> PyResult<()> {
        let (graph, id) = match (&self.graph, self.id) {
            (Some(graph), Some(id)) => (graph, id),
            _ => {
                return Err(PyValueError::new_err(
                    "cannot backpropagate through a detached tensor",
                ));
            }
        };
        let gradients = graph.tape.borrow().backward(id).map_err(value_error)?;
        *graph.gradients.borrow_mut() = gradients;
        Ok(())
    }

    #[getter]
    fn grad(&self) -> Option<Self> {
        let graph = self.graph.as_ref()?;
        let gradient = graph.gradients.borrow().get(&self.id?).cloned()?;
        Some(Self::eager(gradient))
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

    fn __repr__(&self) -> PyResult<String> {
        let value = self.value()?;
        Ok(format!(
            "Tensor(shape={:?}, data={:?})",
            value.shape(),
            value.data()
        ))
    }
}

#[pyclass(name = "Module", subclass, module = "glassbox.nn")]
pub struct PyModuleBase;

#[pyfunction]
#[pyo3(signature = (left, right, atol = 1e-8, rtol = 1e-5))]
fn isclose(left: &PyTensor, right: &PyTensor, atol: f32, rtol: f32) -> PyResult<bool> {
    left.value()?
        .is_close(&right.value()?, atol, rtol)
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
        input.relu()
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
        input.sigmoid()
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
        input.tanh()
    }
}

fn parameters(graph: &Rc<PythonGraph>, ids: Vec<TensorId>) -> PyResult<Vec<PyTensor>> {
    ids.into_iter()
        .map(|id| PyTensor::from_graph(Rc::clone(graph), id))
        .collect()
}

#[pyclass(unsendable, name = "Linear", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyLinear {
    graph: Rc<PythonGraph>,
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
        let graph = default_graph();
        let mut initializer = Initializer::new(seed);
        let layer = RustLinear::new(
            &mut graph.tape.borrow_mut(),
            input_features,
            output_features,
            &mut initializer,
        )
        .map_err(value_error)?;
        graph.register_parameters(&layer.parameters());
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self { graph, layer }))
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input = input.id_on(&self.graph)?;
        let output = self
            .layer
            .forward(&mut self.graph.tape.borrow_mut(), input)
            .map_err(value_error)?;
        PyTensor::from_graph(Rc::clone(&self.graph), output)
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        parameters(&self.graph, self.layer.parameters())
    }
}

#[pyclass(unsendable, name = "LayerNorm", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyLayerNorm {
    graph: Rc<PythonGraph>,
    layer: RustLayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (features, epsilon = 1e-5))]
    fn new(features: usize, epsilon: f32) -> PyResult<PyClassInitializer<Self>> {
        let graph = default_graph();
        let layer = RustLayerNorm::new(&mut graph.tape.borrow_mut(), features, epsilon)
            .map_err(value_error)?;
        graph.register_parameters(&layer.parameters());
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self { graph, layer }))
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let input = input.id_on(&self.graph)?;
        let output = self
            .layer
            .forward(&mut self.graph.tape.borrow_mut(), input)
            .map_err(value_error)?;
        PyTensor::from_graph(Rc::clone(&self.graph), output)
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        parameters(&self.graph, self.layer.parameters())
    }
}

#[pyclass(unsendable, name = "Embedding", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyEmbedding {
    graph: Rc<PythonGraph>,
    layer: RustEmbedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (entries, features, seed = 1337))]
    fn new(entries: usize, features: usize, seed: u64) -> PyResult<PyClassInitializer<Self>> {
        let graph = default_graph();
        let mut initializer = Initializer::new(seed);
        let layer = RustEmbedding::new(
            &mut graph.tape.borrow_mut(),
            entries,
            features,
            &mut initializer,
        )
        .map_err(value_error)?;
        graph.register_parameters(&layer.parameters());
        Ok(PyClassInitializer::from(PyModuleBase).add_subclass(Self { graph, layer }))
    }

    fn forward(&self, indices: Vec<usize>) -> PyResult<PyTensor> {
        let output = self
            .layer
            .forward(&mut self.graph.tape.borrow_mut(), indices)
            .map_err(value_error)?;
        PyTensor::from_graph(Rc::clone(&self.graph), output)
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        parameters(&self.graph, self.layer.parameters())
    }
}

fn optimizer_parameters(
    py: Python<'_>,
    parameters: Vec<Py<PyTensor>>,
) -> PyResult<(Rc<PythonGraph>, Vec<TensorId>)> {
    let first = parameters
        .first()
        .ok_or_else(|| PyValueError::new_err("optimizer requires at least one parameter"))?
        .borrow(py);
    let graph = first
        .graph
        .clone()
        .ok_or_else(|| PyValueError::new_err("optimizer parameters must require gradients"))?;
    drop(first);
    let mut ids = Vec::with_capacity(parameters.len());
    for parameter in parameters {
        let parameter = parameter.borrow(py);
        let current = parameter
            .graph
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("optimizer parameters must require gradients"))?;
        if !Rc::ptr_eq(&graph, current) {
            return Err(PyValueError::new_err(
                "optimizer parameters must belong to the same autograd tape",
            ));
        }
        ids.push(parameter.id.expect("graph tensors have ids"));
    }
    Ok((graph, ids))
}

#[pyclass(unsendable, name = "Adam", module = "glassbox.optim")]
pub struct PyAdam {
    graph: Rc<PythonGraph>,
    parameters: Vec<TensorId>,
    optimizer: Adam,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (parameters, learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weight_decay = 0.0))]
    fn new(
        py: Python<'_>,
        parameters: Vec<Py<PyTensor>>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> PyResult<Self> {
        if weight_decay < 0.0 {
            return Err(PyValueError::new_err("weight_decay must be non-negative"));
        }
        let (graph, parameters) = optimizer_parameters(py, parameters)?;
        Ok(Self {
            graph,
            parameters,
            optimizer: Adam::new(learning_rate, beta1, beta2, epsilon)
                .with_weight_decay(weight_decay),
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.optimizer
            .step(
                &mut self.graph.tape.borrow_mut(),
                &self.parameters,
                &self.graph.gradients.borrow(),
            )
            .map_err(value_error)
    }

    fn zero_grad(&self) -> PyResult<()> {
        self.optimizer
            .zero_grad(&mut self.graph.gradients.borrow_mut());
        self.graph.reset()
    }
}

#[pyclass(unsendable, name = "SGD", module = "glassbox.optim")]
pub struct PySgd {
    graph: Rc<PythonGraph>,
    parameters: Vec<TensorId>,
    optimizer: Sgd,
}

#[pymethods]
impl PySgd {
    #[new]
    #[pyo3(signature = (parameters, learning_rate = 1e-2))]
    fn new(py: Python<'_>, parameters: Vec<Py<PyTensor>>, learning_rate: f32) -> PyResult<Self> {
        let (graph, parameters) = optimizer_parameters(py, parameters)?;
        Ok(Self {
            graph,
            parameters,
            optimizer: Sgd::new(learning_rate),
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.optimizer
            .step(
                &mut self.graph.tape.borrow_mut(),
                &self.parameters,
                &self.graph.gradients.borrow(),
            )
            .map_err(value_error)
    }

    fn zero_grad(&self) -> PyResult<()> {
        self.optimizer
            .zero_grad(&mut self.graph.gradients.borrow_mut());
        self.graph.reset()
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

struct GptState {
    model: Gpt,
    gradients: HashMap<TensorId, Tensor>,
}

#[pyclass(unsendable, name = "Loss", module = "glassbox")]
pub struct PyGptLoss {
    state: Rc<RefCell<GptState>>,
    id: TensorId,
    value: f32,
}

impl PyGptLoss {
    fn backward_inner(&self) -> PyResult<Vec<f32>> {
        let mut state = self.state.borrow_mut();
        let gradients = state.model.backward(self.id).map_err(value_error)?;
        let norms = gradient_norms(&state.model.parameters(), &gradients);
        state.gradients = gradients;
        Ok(norms)
    }
}

#[pymethods]
impl PyGptLoss {
    #[getter]
    fn value(&self) -> f32 {
        self.value
    }

    fn backward(&self) -> PyResult<Vec<f32>> {
        self.backward_inner()
    }

    fn __float__(&self) -> f32 {
        self.value
    }

    fn __repr__(&self) -> String {
        format!("Loss({})", self.value)
    }
}

#[pyclass(unsendable, name = "GPT", extends = PyModuleBase, module = "glassbox.nn")]
pub struct PyGpt {
    state: Rc<RefCell<GptState>>,
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
            state: Rc::new(RefCell::new(GptState {
                model,
                gradients: HashMap::new(),
            })),
            optimizer,
            visualizer: None,
            step: 0,
            trace_interval: 1,
        }))
    }

    fn forward(&mut self, tokens: Vec<Vec<usize>>) -> PyResult<PyTensor> {
        let output = self
            .state
            .borrow_mut()
            .model
            .forward(&tokens)
            .map_err(value_error)?;
        Ok(PyTensor::eager(output))
    }

    #[pyo3(signature = (prompt, max_new_tokens, temperature = 1.0, top_k = None, seed = 1337))]
    fn generate(
        &mut self,
        prompt: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        seed: u64,
    ) -> PyResult<Vec<usize>> {
        self.state
            .borrow_mut()
            .model
            .generate(&prompt, max_new_tokens, temperature, top_k, seed)
            .map_err(value_error)
    }

    fn loss(&mut self, tokens: Vec<Vec<usize>>, targets: Vec<Vec<usize>>) -> PyResult<PyGptLoss> {
        let mut state = self.state.borrow_mut();
        state.model.reset_tape().map_err(value_error)?;
        let id = state.model.loss(&tokens, &targets).map_err(value_error)?;
        let value = state.model.loss_value(id).map_err(value_error)?;
        state.gradients.clear();
        drop(state);
        Ok(PyGptLoss {
            state: Rc::clone(&self.state),
            id,
            value,
        })
    }

    fn backward(&self, loss: &PyGptLoss) -> PyResult<Vec<f32>> {
        if !Rc::ptr_eq(&self.state, &loss.state) {
            return Err(PyValueError::new_err(
                "loss was created by a different GPT model",
            ));
        }
        loss.backward_inner()
    }

    #[pyo3(signature = (maximum_norm = 1.0))]
    fn step(&mut self, maximum_norm: f32) -> PyResult<()> {
        let mut state = self.state.borrow_mut();
        let parameters = state.model.parameters();
        clip_gradients(&parameters, &mut state.gradients, maximum_norm).map_err(value_error)?;
        let GptState { model, gradients } = &mut *state;
        self.optimizer
            .step(model.tape_mut(), &parameters, gradients)
            .map_err(value_error)?;
        self.step += 1;
        Ok(())
    }

    fn zero_grad(&self) {
        self.optimizer
            .zero_grad(&mut self.state.borrow_mut().gradients);
    }

    #[pyo3(signature = (tokens, targets, maximum_norm = 1.0))]
    fn train_step(
        &mut self,
        tokens: Vec<Vec<usize>>,
        targets: Vec<Vec<usize>>,
        maximum_norm: f32,
    ) -> PyResult<(f32, Vec<f32>)> {
        let traced = self.visualizer.is_some() && self.step.is_multiple_of(self.trace_interval);
        self.state.borrow_mut().model.set_trace(self.step, traced);
        let result: PyResult<(f32, Vec<f32>)> = (|| {
            let loss = self.loss(tokens, targets)?;
            let value = loss.value;
            let norms = loss.backward_inner()?;
            self.step(maximum_norm)?;
            self.zero_grad();
            self.state
                .borrow_mut()
                .model
                .reset_tape()
                .map_err(value_error)?;
            Ok((value, norms))
        })();
        self.state.borrow_mut().model.set_trace(self.step, false);
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
        self.state
            .borrow_mut()
            .model
            .set_event_emitter(Some(server.emitter()));
        self.trace_interval = trace_interval;
        self.visualizer = Some(server);
        Ok(url)
    }

    fn disable_visualizer(&mut self) {
        let mut state = self.state.borrow_mut();
        state.model.set_event_emitter(None);
        state.model.set_trace(self.step, false);
        self.visualizer = None;
    }

    fn dropped_events(&self) -> u64 {
        self.visualizer
            .as_ref()
            .map(VisualizerServer::dropped)
            .unwrap_or(0)
    }

    fn evaluate(&mut self, tokens: Vec<Vec<usize>>, targets: Vec<Vec<usize>>) -> PyResult<f32> {
        let mut state = self.state.borrow_mut();
        state.model.reset_tape().map_err(value_error)?;
        let loss = state.model.loss(&tokens, &targets).map_err(value_error)?;
        let value = state.model.loss_value(loss).map_err(value_error)?;
        state.model.reset_tape().map_err(value_error)?;
        Ok(value)
    }

    fn save_checkpoint(&self, path: &str) -> PyResult<()> {
        self.state
            .borrow()
            .model
            .save_checkpoint(path)
            .map_err(value_error)
    }

    fn load_checkpoint(&mut self, path: &str) -> PyResult<()> {
        self.state
            .borrow_mut()
            .model
            .load_checkpoint(path)
            .map_err(value_error)
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(self
            .state
            .borrow()
            .model
            .parameter_values()
            .map_err(value_error)?
            .into_iter()
            .map(PyTensor::eager)
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

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        self.dataset.encode(text).map_err(value_error)
    }

    fn decode(&self, tokens: Vec<usize>) -> PyResult<String> {
        self.dataset.decode(&tokens).map_err(value_error)
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
    module.add_class::<PyGptLoss>()?;
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
    py.import("sys")?
        .getattr("modules")?
        .set_item("glassbox.nn", &nn)?;
    let data = PyModule::new(py, "glassbox.data")?;
    data.add_class::<PyCharDataset>()?;
    module.add_submodule(&data)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("glassbox.data", &data)?;
    let optim = PyModule::new(py, "glassbox.optim")?;
    optim.add_class::<PyAdam>()?;
    optim.add_class::<PySgd>()?;
    module.add_submodule(&optim)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("glassbox.optim", &optim)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
