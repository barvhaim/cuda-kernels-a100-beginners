#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

// One thread computes one output dimension: y[out] = sum_in x[in] * W[in,out].
__global__ void linear_projection(const float* x, const float* weight, float* y,
                                  int input_size, int output_size) {
  const int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (out < output_size) {
    float value = 0.0f;
    for (int in = 0; in < input_size; ++in) {
      value += x[in] * weight[in * output_size + out];
    }
    y[out] = value;
  }
}

int main() {
  constexpr int input_size = 3;
  constexpr int output_size = 2;
  const std::vector<float> x{1, 2, 3};
  const std::vector<float> weight{
      1, 0,
      0, 1,
      1, 1,
  };
  std::vector<float> y(output_size);
  float *d_x = nullptr, *d_weight = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, y.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));
  linear_projection<<<1, 4>>>(d_x, d_weight, d_y, input_size, output_size);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_y));
  if (std::fabs(y[0] - 4.0f) > 1e-6f || std::fabs(y[1] - 5.0f) > 1e-6f) return 1;
  std::cout << "PASS linear_projection: [1,2,3] projected to [4,5]\n";
  return 0;
}
