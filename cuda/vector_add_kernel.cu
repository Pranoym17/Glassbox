extern "C" __global__ void vector_add(const float* a, const float* b, float* c, unsigned long long n) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
