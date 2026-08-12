#include "cuda_check.cuh"

#include <iostream>
#include <vector>

__global__ void add_one(int* values, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    values[i] += 1;
  }
}

int main() {
  std::vector<int> host_values{10, 20, 30, 40};
  const int n = static_cast<int>(host_values.size());
  const size_t bytes = host_values.size() * sizeof(int);

  int* device_values = nullptr;
  CUDA_CHECK(cudaMalloc(&device_values, bytes));
  CUDA_CHECK(cudaMemcpy(device_values, host_values.data(), bytes,
                        cudaMemcpyHostToDevice));

  add_one<<<1, n>>>(device_values, n);
  check_kernel();

  CUDA_CHECK(cudaMemcpy(host_values.data(), device_values, bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_values));

  const std::vector<int> expected{11, 21, 31, 41};
  if (host_values != expected) {
    std::cerr << "FAIL memory_roundtrip\n";
    return 1;
  }
  std::cout << "PASS memory_roundtrip: 11 21 31 41\n";
  return 0;
}
