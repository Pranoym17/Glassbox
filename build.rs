use std::{env, path::PathBuf, process::Command};

fn compile_ptx(source: &str, output_name: &str) {
    println!("cargo:rerun-if-changed={source}");
    let output = PathBuf::from(env::var("OUT_DIR").unwrap()).join(output_name);
    let status = Command::new("nvcc")
        .args(["--ptx", "-O2", source, "-o"])
        .arg(&output)
        .status()
        .expect("nvcc must be available on PATH");
    assert!(status.success(), "nvcc failed to compile {source}");
}

fn main() {
    compile_ptx("cuda/vector_add_kernel.cu", "vector_add.ptx");
    compile_ptx("cuda/tensor_ops.cu", "tensor_ops.ptx");
}
