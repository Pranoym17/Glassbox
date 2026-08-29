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

// Naive matrix multiplication: one CUDA thread computes C[row, col].
extern "C" __global__ void matmul_naive(
    const float* left, const float* right, float* output,
    unsigned long long rows, unsigned long long inner, unsigned long long columns
) {
    const unsigned long long column = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row >= rows || column >= columns) return;

    float sum = 0.0f;
    for (unsigned long long index = 0; index < inner; ++index) {
        sum += left[row * inner + index] * right[index * columns + column];
    }
    output[row * columns + column] = sum;
}

extern "C" __global__ void softmax_rows(
    const float* input, float* output, unsigned long long rows, unsigned long long columns
) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    if (row >= rows) return;

    float local_max = -3.402823466e+38F;
    for (unsigned long long column = thread; column < columns; column += blockDim.x) {
        local_max = fmaxf(local_max, input[row * columns + column]);
    }
    shared[thread] = local_max;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
        if (thread < width) shared[thread] = fmaxf(shared[thread], shared[thread + width]);
        __syncthreads();
    }
    const float maximum = shared[0];

    float local_sum = 0.0f;
    for (unsigned long long column = thread; column < columns; column += blockDim.x) {
        local_sum += expf(input[row * columns + column] - maximum);
    }
    shared[thread] = local_sum;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
        if (thread < width) shared[thread] += shared[thread + width];
        __syncthreads();
    }
    const float sum = shared[0];
    for (unsigned long long column = thread; column < columns; column += blockDim.x) {
        output[row * columns + column] = expf(input[row * columns + column] - maximum) / sum;
    }
}

extern "C" __global__ void layer_norm_rows(
    const float* input, const float* gamma, const float* beta, float* output,
    unsigned long long rows, unsigned long long columns, float epsilon
) {
    extern __shared__ float shared[];
    const unsigned int thread = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    if (row >= rows) return;

    float local_sum = 0.0f;
    for (unsigned long long column = thread; column < columns; column += blockDim.x) {
        local_sum += input[row * columns + column];
    }
    shared[thread] = local_sum;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
        if (thread < width) shared[thread] += shared[thread + width];
        __syncthreads();
    }
    const float mean = shared[0] / static_cast<float>(columns);

    float local_variance = 0.0f;
    for (unsigned long long column = thread; column < columns; column += blockDim.x) {
        const float centered = input[row * columns + column] - mean;
        local_variance += centered * centered;
    }
    shared[thread] = local_variance;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
        if (thread < width) shared[thread] += shared[thread + width];
        __syncthreads();
    }
    const float inverse_std = rsqrtf(shared[0] / static_cast<float>(columns) + epsilon);
    for (unsigned long long column = thread; column < columns; column += blockDim.x) {
        const float normalized = (input[row * columns + column] - mean) * inverse_std;
        output[row * columns + column] = normalized * gamma[column] + beta[column];
    }
}
