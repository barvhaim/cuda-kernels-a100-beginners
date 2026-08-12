#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

__host__ __device__ float silu_value(float x) {
  // Stable sigmoid: never evaluate exp(-x) when x is very negative.
  const float sigmoid =
      x >= 0.0f ? 1.0f / (1.0f + expf(-x))
                : expf(x) / (1.0f + expf(x));
  return x * sigmoid;
}

// LLM connection: SiLU is commonly used inside gated MLP blocks.
__global__ void silu(const float* input, float* output, int elements) {
  const int start = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = start; i < elements; i += stride) output[i] = silu_value(input[i]);
}

int main() {
  const std::vector<float> input{-100.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  std::vector<float> output(input.size());
  const size_t bytes = input.size() * sizeof(float);
  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  silu<<<1, 4>>>(d_input, d_output, static_cast<int>(input.size()));
  check_kernel();
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  for (size_t i = 0; i < input.size(); ++i) {
    if (!std::isfinite(output[i]) ||
        std::fabs(output[i] - silu_value(input[i])) > 1e-6f) return 1;
  }
  std::cout << "PASS silu_activation: grid-stride loop covered the MLP vector\n";
  return 0;
}
