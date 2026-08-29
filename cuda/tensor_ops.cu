extern "C" __global__ void add(const float* a, const float* b, float* out, unsigned long long n) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" __global__ void mul(const float* a, const float* b, float* out, unsigned long long n) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}
extern "C" __global__ void sub(const float* a, const float* b, float* out, unsigned long long n) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}
extern "C" __global__ void div_kernel(const float* a, const float* b, float* out, unsigned long long n) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] / b[i];
}
extern "C" __global__ void exp_kernel(const float* input, float* out, unsigned long long n) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(input[i]);
}
extern "C" __global__ void log_kernel(const float* input, float* out, unsigned long long n) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = logf(input[i]);
}

// Stage 1: each (output index, chunk) block reduces one 256-value axis segment.
extern "C" __global__ void sum_axis_partials(
    const float* input, float* partials, unsigned long long outer,
    unsigned long long reduce_size, unsigned long long inner, unsigned long long chunks
) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const unsigned long long output_index = blockIdx.x;
    const unsigned long long chunk = blockIdx.y;
    const unsigned long long outer_index = output_index / inner;
    const unsigned long long inner_index = output_index % inner;
    const unsigned long long reduce_index = chunk * blockDim.x + thread;
    const unsigned long long base = outer_index * reduce_size * inner + inner_index;

    shared[thread] = reduce_index < reduce_size ? input[base + reduce_index * inner] : 0.0f;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
        if (thread < width) shared[thread] += shared[thread + width];
        __syncthreads();
    }
    if (thread == 0) partials[output_index * chunks + chunk] = shared[0];
}

// Stage 2: one block combines the partial sums for one output element.
extern "C" __global__ void sum_axis_finalize(
    const float* partials, float* output, unsigned long long output_count, unsigned long long chunks
) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const unsigned long long output_index = blockIdx.x;
    if (output_index >= output_count) return;

    float sum = 0.0f;
    for (unsigned long long chunk = thread; chunk < chunks; chunk += blockDim.x) {
        sum += partials[output_index * chunks + chunk];
    }
    shared[thread] = sum;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
        if (thread < width) shared[thread] += shared[thread + width];
        __syncthreads();
    }
    if (thread == 0) output[output_index] = shared[0];
}
