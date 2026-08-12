#include "cuda_check.cuh"

#include <cstdio>
#include <iostream>

__global__ void show_global_index() {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("block=%d thread=%d global=%d\n", blockIdx.x, threadIdx.x, i);
}

int main() {
  constexpr int blocks = 2;
  constexpr int threads = 4;
  show_global_index<<<blocks, threads>>>();
  check_kernel();
  std::cout << "PASS global_index: expected global indices are 0 through 7\n";
  return 0;
}
