use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=cuda/vector_add_kernel.cu");
    let output = PathBuf::from(env::var("OUT_DIR").unwrap()).join("vector_add.ptx");
    let status = Command::new("nvcc")
        .args(["--ptx", "-O2", "cuda/vector_add_kernel.cu", "-o"])
        .arg(&output)
        .status()
        .expect("nvcc must be available on PATH");
    assert!(status.success(), "nvcc failed to compile vector_add_kernel.cu");
}
