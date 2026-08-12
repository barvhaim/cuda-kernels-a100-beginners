#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

// LLM connection: hidden_out = layer_output + residual_stream.
__global__ void residual_add(const float* layer_output, const float* residual,
                             float* hidden_out, int elements) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elements) hidden_out[i] = layer_output[i] + residual[i];
}

int main() {
  const std::vector<float> layer{0.5f, 1.0f, -1.0f, 2.0f, 3.0f};
  const std::vector<float> residual{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> output(layer.size());
  const size_t bytes = layer.size() * sizeof(float);
  float *d_layer = nullptr, *d_residual = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_layer, bytes));
  CUDA_CHECK(cudaMalloc(&d_residual, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMemcpy(d_layer, layer.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_residual, residual.data(), bytes, cudaMemcpyHostToDevice));

  residual_add<<<1, 8>>>(d_layer, d_residual, d_output, static_cast<int>(layer.size()));
  check_kernel();
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_layer));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_output));

  const std::vector<float> expected{1.5f, 3.0f, 2.0f, 6.0f, 8.0f};
  if (output != expected) return 1;
  std::cout << "PASS residual_add: 8 threads handled 5 hidden values\n";
  return 0;
}
