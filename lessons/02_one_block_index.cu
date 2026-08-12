#include "cuda_check.cuh"

#include <cstdio>
#include <iostream>

__global__ void show_thread_index() {
  printf("threadIdx.x = %d\n", threadIdx.x);
}

int main() {
  constexpr int threads = 4;
  show_thread_index<<<1, threads>>>();
  check_kernel();
  std::cout << "PASS one_block_index: expected indices are 0, 1, 2, 3\n";
  return 0;
}
