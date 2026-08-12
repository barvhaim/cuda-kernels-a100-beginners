#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error__ = (call);                                                \
    if (error__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "    \
                << cudaGetErrorString(error__) << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

inline void check_kernel() {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
