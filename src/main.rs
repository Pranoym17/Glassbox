use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = 1_000_003;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.25).collect();

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let ptx = Ptx::from(include_str!(concat!(env!("OUT_DIR"), "/vector_add.ptx")));
    let module = context.load_module(ptx)?;
    let function = module.load_function("vector_add")?;

    let d_a = stream.clone_htod(&a)?;
    let d_b = stream.clone_htod(&b)?;
    let mut d_result = stream.alloc_zeros::<f32>(n)?;
    let threads_per_block = 256_u32;
    let config = LaunchConfig {
        grid_dim: (
            ((n as u32) + threads_per_block - 1) / threads_per_block,
            1,
            1,
        ),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    // Safe ownership/lifetimes are tracked by cudarc; kernel ABI remains the caller's responsibility.
    unsafe {
        stream
            .launch_builder(&function)
            .arg(&d_a)
            .arg(&d_b)
            .arg(&mut d_result)
            .arg(&(n as u64))
            .launch(config)?;
    }
    let result = stream.clone_dtoh(&d_result)?;

    for (index, (&actual, (&left, &right))) in result.iter().zip(a.iter().zip(&b)).enumerate() {
        if (actual - (left + right)).abs() > 1e-6 {
            return Err(format!("verification failed at index {index}").into());
        }
    }
    println!("PASS: cudarc vector_add verified {n} elements.");
    Ok(())
}
