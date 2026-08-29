use std::{fmt, sync::Arc};

use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

use crate::{Tensor, TensorError};

const THREADS_PER_BLOCK: u32 = 256;
const _: () = assert!(THREADS_PER_BLOCK.is_power_of_two());

pub struct CudaBackend {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

#[derive(Debug)]
pub enum GpuError {
    Driver(DriverError),
    Tensor(TensorError),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "CUDA driver error: {error}"),
            Self::Tensor(error) => write!(f, "tensor error: {error}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl From<TensorError> for GpuError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl CudaBackend {
    pub fn new() -> Result<Self, GpuError> {
        let context = CudaContext::new(0)?;
        let stream = context.default_stream();
        let ptx = Ptx::from(include_str!(concat!(env!("OUT_DIR"), "/tensor_ops.ptx")));
        let module = context.load_module(ptx)?;
        Ok(Self {
            context,
            stream,
            module,
        })
    }

    pub fn device_name(&self) -> Result<String, GpuError> {
        Ok(self.context.name()?)
    }

    pub fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, GpuError> {
        self.binary("add", left, right)
    }

    pub fn mul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, GpuError> {
        self.binary("mul", left, right)
    }

    pub fn sub(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, GpuError> {
        self.binary("sub", left, right)
    }

    pub fn div(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, GpuError> {
        self.binary("div_kernel", left, right)
    }

    pub fn exp(&self, input: &Tensor) -> Result<Tensor, GpuError> {
        self.unary("exp_kernel", input)
    }

    pub fn log(&self, input: &Tensor) -> Result<Tensor, GpuError> {
        self.unary("log_kernel", input)
    }
    pub fn softmax(&self, input: &Tensor) -> Result<Tensor, GpuError> {
        if input.shape().len() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: input.shape().len(),
            }
            .into());
        }
        let input = input.contiguous();
        let (rows, columns) = (input.shape()[0], input.shape()[1]);
        let device_input = self.stream.clone_htod(input.data())?;
        let mut device_output = self.stream.alloc_zeros::<f32>(input.data().len())?;
        let function = self.module.load_function("softmax_rows")?;
        unsafe {
            self.stream
                .launch_builder(&function)
                .arg(&device_input)
                .arg(&mut device_output)
                .arg(&(rows as u64))
                .arg(&(columns as u64))
                .launch(row_config(rows))?;
        }
        Ok(Tensor::from_vec(
            input.shape().to_vec(),
            self.stream.clone_dtoh(&device_output)?,
        )?)
    }

    pub fn layer_norm(
        &self,
        input: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, GpuError> {
        if input.shape().len() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: input.shape().len(),
            }
            .into());
        }
        let input = input.contiguous();
        let (rows, columns) = (input.shape()[0], input.shape()[1]);
        if gamma.shape() != [columns] || beta.shape() != [columns] {
            return Err(TensorError::IncompatibleShapes {
                left: gamma.shape().to_vec(),
                right: beta.shape().to_vec(),
            }
            .into());
        }
        let device_input = self.stream.clone_htod(input.data())?;
        let device_gamma = self.stream.clone_htod(gamma.data())?;
        let device_beta = self.stream.clone_htod(beta.data())?;
        let mut device_output = self.stream.alloc_zeros::<f32>(input.data().len())?;
        let function = self.module.load_function("layer_norm_rows")?;
        unsafe {
            self.stream
                .launch_builder(&function)
                .arg(&device_input)
                .arg(&device_gamma)
                .arg(&device_beta)
                .arg(&mut device_output)
                .arg(&(rows as u64))
                .arg(&(columns as u64))
                .arg(&epsilon)
                .launch(row_config(rows))?;
        }
        Ok(Tensor::from_vec(
            input.shape().to_vec(),
            self.stream.clone_dtoh(&device_output)?,
        )?)
    }

    pub fn matmul(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, GpuError> {
        if left.shape().len() != 2
            || right.shape().len() != 2
            || left.shape()[1] != right.shape()[0]
        {
            return Err(TensorError::IncompatibleMatMul {
                left: left.shape().to_vec(),
                right: right.shape().to_vec(),
            }
            .into());
        }
        let left = left.contiguous();
        let right = right.contiguous();
        let (rows, inner, columns) = (left.shape()[0], left.shape()[1], right.shape()[1]);
        let device_left = self.stream.clone_htod(left.data())?;
        let device_right = self.stream.clone_htod(right.data())?;
        let mut device_output = self.stream.alloc_zeros::<f32>(rows * columns)?;
        let function = self.module.load_function("matmul_naive")?;
        let tile = 16_u32;
        unsafe {
            self.stream
                .launch_builder(&function)
                .arg(&device_left)
                .arg(&device_right)
                .arg(&mut device_output)
                .arg(&(rows as u64))
                .arg(&(inner as u64))
                .arg(&(columns as u64))
                .launch(LaunchConfig {
                    grid_dim: (
                        (columns as u32).div_ceil(tile),
                        (rows as u32).div_ceil(tile),
                        1,
                    ),
                    block_dim: (tile, tile, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(Tensor::from_vec(
            vec![rows, columns],
            self.stream.clone_dtoh(&device_output)?,
        )?)
    }

    pub fn sum_axis(&self, input: &Tensor, axis: usize) -> Result<Tensor, GpuError> {
        if axis >= input.shape().len() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: input.shape().len(),
            }
            .into());
        }
        let input = input.contiguous();
        let outer = input.shape()[..axis].iter().product::<usize>();
        let reduce_size = input.shape()[axis];
        let inner = input.shape()[axis + 1..].iter().product::<usize>();
        let output_count = outer * inner;
        let chunks = reduce_size.div_ceil(THREADS_PER_BLOCK as usize);
        let mut output_shape = input.shape().to_vec();
        output_shape.remove(axis);

        let device_input = self.stream.clone_htod(input.data())?;
        let mut device_partials = self.stream.alloc_zeros::<f32>(output_count * chunks)?;
        let mut device_output = self.stream.alloc_zeros::<f32>(output_count)?;
        let shared_memory_bytes = THREADS_PER_BLOCK * std::mem::size_of::<f32>() as u32;

        let partial_kernel = self.module.load_function("sum_axis_partials")?;
        unsafe {
            self.stream
                .launch_builder(&partial_kernel)
                .arg(&device_input)
                .arg(&mut device_partials)
                .arg(&(outer as u64))
                .arg(&(reduce_size as u64))
                .arg(&(inner as u64))
                .arg(&(chunks as u64))
                .launch(LaunchConfig {
                    grid_dim: (output_count as u32, chunks as u32, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: shared_memory_bytes,
                })?;
        }

        let final_kernel = self.module.load_function("sum_axis_finalize")?;
        unsafe {
            self.stream
                .launch_builder(&final_kernel)
                .arg(&device_partials)
                .arg(&mut device_output)
                .arg(&(output_count as u64))
                .arg(&(chunks as u64))
                .launch(LaunchConfig {
                    grid_dim: (output_count as u32, 1, 1),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: shared_memory_bytes,
                })?;
        }
        Ok(Tensor::from_vec(
            output_shape,
            self.stream.clone_dtoh(&device_output)?,
        )?)
    }

    fn binary(&self, kernel_name: &str, left: &Tensor, right: &Tensor) -> Result<Tensor, GpuError> {
        let shape = Tensor::broadcast_shape(left.shape(), right.shape())?;
        // Broadcasting is materialized as contiguous inputs before the simple
        // one-thread-per-element CUDA kernel. Strided GPU views come next.
        let left = left.broadcast_to(&shape)?;
        let right = right.broadcast_to(&shape)?;
        let device_left = self.stream.clone_htod(left.data())?;
        let device_right = self.stream.clone_htod(right.data())?;
        let mut device_output = self.stream.alloc_zeros::<f32>(left.data().len())?;
        let function = self.module.load_function(kernel_name)?;
        unsafe {
            self.stream
                .launch_builder(&function)
                .arg(&device_left)
                .arg(&device_right)
                .arg(&mut device_output)
                .arg(&(left.data().len() as u64))
                .launch(elementwise_config(left.data().len()))?;
        }
        Ok(Tensor::from_vec(
            shape,
            self.stream.clone_dtoh(&device_output)?,
        )?)
    }

    fn unary(&self, kernel_name: &str, input: &Tensor) -> Result<Tensor, GpuError> {
        let input = input.contiguous();
        let device_input = self.stream.clone_htod(input.data())?;
        let mut device_output = self.stream.alloc_zeros::<f32>(input.data().len())?;
        let function = self.module.load_function(kernel_name)?;
        unsafe {
            self.stream
                .launch_builder(&function)
                .arg(&device_input)
                .arg(&mut device_output)
                .arg(&(input.data().len() as u64))
                .launch(elementwise_config(input.data().len()))?;
        }
        Ok(Tensor::from_vec(
            input.shape().to_vec(),
            self.stream.clone_dtoh(&device_output)?,
        )?)
    }
}

fn elementwise_config(count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((count as u32).div_ceil(THREADS_PER_BLOCK), 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn row_config(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: THREADS_PER_BLOCK * std::mem::size_of::<f32>() as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::CudaBackend;
    use crate::Tensor;

    fn assert_close(gpu: &Tensor, cpu: &Tensor) {
        assert_eq!(gpu.shape(), cpu.shape());
        for (index, (&actual, &expected)) in gpu.data().iter().zip(cpu.data()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "mismatch at {index}: GPU={actual}, CPU={expected}"
            );
        }
    }

    fn operands() -> (Tensor, Tensor) {
        (
            Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Tensor::from_vec(vec![3], vec![0.5, 1.5, 2.5]).unwrap(),
        )
    }

    #[test]
    fn gpu_elementwise_kernels_match_cpu() {
        let backend = CudaBackend::new().unwrap();
        let (left, right) = operands();
        assert_close(
            &backend.add(&left, &right).unwrap(),
            &left.add(&right).unwrap(),
        );
        assert_close(
            &backend.mul(&left, &right).unwrap(),
            &left.mul(&right).unwrap(),
        );
        assert_close(
            &backend.sub(&left, &right).unwrap(),
            &left.sub(&right).unwrap(),
        );
        assert_close(
            &backend.div(&left, &right).unwrap(),
            &left.div(&right).unwrap(),
        );
        assert_close(&backend.exp(&left).unwrap(), &left.exp().unwrap());
        assert_close(&backend.log(&left).unwrap(), &left.log().unwrap());
    }

    #[test]
    fn gpu_reduction_matches_cpu_for_each_axis() {
        let backend = CudaBackend::new().unwrap();
        let input = Tensor::from_vec(
            vec![3, 4, 5],
            (1..=60).map(|value| value as f32 * 0.1).collect(),
        )
        .unwrap();
        assert_close(
            &backend.sum_axis(&input, 0).unwrap(),
            &input.sum_axis(0).unwrap(),
        );
        assert_close(
            &backend.sum_axis(&input, 1).unwrap(),
            &input.sum_axis(1).unwrap(),
        );
        assert_close(
            &backend.sum_axis(&input, 2).unwrap(),
            &input.sum_axis(2).unwrap(),
        );
    }
    #[test]
    fn gpu_reduction_combines_multiple_block_partials() {
        let backend = CudaBackend::new().unwrap();
        let input = Tensor::from_vec(
            vec![2, 1_025, 3],
            (0..6_150)
                .map(|value| (value % 31) as f32 * 0.125)
                .collect(),
        )
        .unwrap();

        assert_close(
            &backend.sum_axis(&input, 1).unwrap(),
            &input.sum_axis(1).unwrap(),
        );
    }
    #[test]
    fn gpu_naive_matmul_matches_cpu_for_4x4_and_16x16() {
        let backend = CudaBackend::new().unwrap();
        for size in [4, 16] {
            let left = Tensor::from_vec(
                vec![size, size],
                (0..size * size)
                    .map(|index| (index % 7) as f32 - 3.0)
                    .collect(),
            )
            .unwrap();
            let right = Tensor::from_vec(
                vec![size, size],
                (0..size * size)
                    .map(|index| (index % 5) as f32 * 0.25)
                    .collect(),
            )
            .unwrap();
            assert_close(
                &backend.matmul(&left, &right).unwrap(),
                &left.matmul(&right).unwrap(),
            );
        }
    }

    #[test]
    fn gpu_softmax_and_layer_norm_match_cpu() {
        use crate::transformer::{layer_norm, softmax};

        let backend = CudaBackend::new().unwrap();
        for columns in [8, 513] {
            let rows = 3;
            let input = Tensor::from_vec(
                vec![rows, columns],
                (0..rows * columns)
                    .map(|index| (index % 37) as f32 * 0.07 - 1.1)
                    .collect(),
            )
            .unwrap();
            let gamma = Tensor::from_vec(
                vec![columns],
                (0..columns)
                    .map(|index| 0.7 + (index % 13) as f32 * 0.02)
                    .collect(),
            )
            .unwrap();
            let beta = Tensor::from_vec(
                vec![columns],
                (0..columns)
                    .map(|index| (index % 17) as f32 * 0.01 - 0.08)
                    .collect(),
            )
            .unwrap();

            assert_close(&backend.softmax(&input).unwrap(), &softmax(&input).unwrap());
            assert_close(
                &backend.layer_norm(&input, &gamma, &beta, 1e-5).unwrap(),
                &layer_norm(&input, &gamma, &beta, 1e-5).unwrap().0,
            );
        }
    }
}
