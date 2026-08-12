#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

// Educational one-block RMSNorm for one hidden vector.
// Power-of-two threads are required by the reduction below.
__global__ void rmsnorm(const float* input, const float* weight, float* output,
                        int hidden_size, float epsilon) {
  extern __shared__ float squares[];
  const int tid = threadIdx.x;
  float x = tid < hidden_size ? input[tid] : 0.0f;
  squares[tid] = x * x;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) squares[tid] += squares[tid + offset];
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(squares[0] / hidden_size + epsilon);
  if (tid < hidden_size) output[tid] = x * inverse_rms * weight[tid];
}

int main() {
  constexpr int hidden_size = 4;
  constexpr int threads = 4;
  static_assert((threads & (threads - 1)) == 0, "threads must be a power of two");
  static_assert(threads >= hidden_size,
                "educational one-block RMSNorm needs one thread per hidden dimension");
  const std::vector<float> input{1, 2, 3, 4};
  const std::vector<float> weight(hidden_size, 1.0f);
  std::vector<float> output(hidden_size);
  const size_t bytes = hidden_size * sizeof(float);
  float *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_weight, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), bytes, cudaMemcpyHostToDevice));
  rmsnorm<<<1, threads, threads * sizeof(float)>>>(d_input, d_weight, d_output, hidden_size, 1e-5f);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_output));

  float sum_squares = 0.0f;
  for (float x : input) sum_squares += x * x;
  const float inverse_rms = 1.0f / std::sqrt(sum_squares / hidden_size + 1e-5f);
  for (int i = 0; i < hidden_size; ++i) {
    if (std::fabs(output[i] - input[i] * inverse_rms) > 1e-5f) return 1;
  }
  std::cout << "PASS rmsnorm: reduction normalized one hidden state\n";
  return 0;
}
