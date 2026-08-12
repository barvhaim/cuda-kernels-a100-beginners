#include "cuda_check.cuh"

#include <iostream>

int main() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  if (count == 0) {
    std::cerr << "No CUDA device found\n";
    return 1;
  }

  for (int device = 0; device < count; ++device) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    std::cout << "Device " << device << ": " << p.name << '\n'
              << "  compute capability: " << p.major << '.' << p.minor << '\n'
              << "  SMs: " << p.multiProcessorCount << '\n'
              << "  warp size: " << p.warpSize << '\n'
              << "  max threads/block: " << p.maxThreadsPerBlock << '\n'
              << "  global memory: " << p.totalGlobalMem / (1024 * 1024) << " MiB\n";
  }
  return 0;
}
