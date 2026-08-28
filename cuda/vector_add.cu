#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { const cudaError_t status = (call); if (status != cudaSuccess) { std::cerr << #call << " failed: " << cudaGetErrorString(status) << '\n'; std::exit(EXIT_FAILURE); } } while (false)

// One thread produces one output element.
__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    constexpr size_t n = 1'000'003;  // Deliberately not divisible by block size.
    constexpr unsigned int threads_per_block = 256;
    const unsigned int blocks = static_cast<unsigned int>((n + threads_per_block - 1) / threads_per_block);
    std::vector<float> a(n), b(n), result(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(n - i) * 0.25f;
    }

    float *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    const size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_result, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_result, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), d_result, bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(result[i] - (a[i] + b[i])) > 1e-6f) {
            std::cerr << "Verification failed at index " << i << '\n';
            return EXIT_FAILURE;
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    std::cout << "PASS: vector_add verified " << n << " elements using " << blocks
              << " blocks x " << threads_per_block << " threads.\n";
}
