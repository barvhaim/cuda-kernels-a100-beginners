#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

// One block handles one token; each thread copies one hidden dimension.
__global__ void token_embedding(const int* token_ids, const float* table,
                                float* output, int tokens, int vocab_size,
                                int hidden_size) {
  const int token_position = blockIdx.x;
  const int hidden_dimension = threadIdx.x;
  if (token_position < tokens && hidden_dimension < hidden_size) {
    const int token_id = token_ids[token_position];
    output[token_position * hidden_size + hidden_dimension] =
        (token_id >= 0 && token_id < vocab_size)
            ? table[token_id * hidden_size + hidden_dimension]
            : 0.0f;
  }
}

int main() {
  constexpr int vocab_size = 4;
  constexpr int hidden_size = 3;
  constexpr int tokens = 2;
  static_assert(hidden_size <= 1024,
                "educational kernel maps one hidden vector to one block");
  const std::vector<int> token_ids{2, 0};
  const std::vector<float> table{
      0, 1, 2,  // token 0
      3, 4, 5,  // token 1
      6, 7, 8,  // token 2
      9, 10, 11 // token 3
  };
  std::vector<float> output(tokens * hidden_size);
  for (int token_id : token_ids) {
    if (token_id < 0 || token_id >= vocab_size) {
      std::cerr << "Token ID out of vocabulary range\n";
      return 1;
    }
  }
  int* d_tokens = nullptr;
  float *d_table = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tokens, tokens * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_table, vocab_size * hidden_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_tokens, token_ids.data(), tokens * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));

  token_embedding<<<tokens, hidden_size>>>(d_tokens, d_table, d_output, tokens,
                                            vocab_size, hidden_size);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_tokens));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_output));

  const std::vector<float> expected{6, 7, 8, 0, 1, 2};
  if (output != expected) return 1;
  std::cout << "PASS token_embedding: token IDs 2,0 selected two embedding rows\n";
  return 0;
}
