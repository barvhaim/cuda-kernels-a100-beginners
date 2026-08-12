#include "cuda_check.cuh"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void embedding_lookup(int token_id, const float* table, float* hidden, int hidden_size) {
  const int i = threadIdx.x;
  if (i < hidden_size) hidden[i] = table[token_id * hidden_size + i];
}

__global__ void residual_add(const float* update, float* hidden, int hidden_size) {
  const int i = threadIdx.x;
  if (i < hidden_size) hidden[i] += update[i];
}

__global__ void rmsnorm(const float* hidden, float* normalized, int hidden_size) {
  extern __shared__ float squares[];
  const int tid = threadIdx.x;
  const float x = tid < hidden_size ? hidden[tid] : 0.0f;
  squares[tid] = x * x;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) squares[tid] += squares[tid + offset];
    __syncthreads();
  }
  if (tid < hidden_size) normalized[tid] = x * rsqrtf(squares[0] / hidden_size + 1e-5f);
}

__global__ void project(const float* hidden, const float* weight, float* logits,
                        int hidden_size, int vocab_size) {
  const int token = threadIdx.x;
  if (token < vocab_size) {
    float value = 0.0f;
    for (int i = 0; i < hidden_size; ++i) value += hidden[i] * weight[i * vocab_size + token];
    logits[token] = value;
  }
}

int main() {
  constexpr int hidden_size = 4;
  constexpr int vocab_size = 3;
  constexpr int token_id = 1;
  static_assert(hidden_size <= 1024 && vocab_size <= 1024,
                "educational pipeline uses one block per vector operation");
  static_assert((hidden_size & (hidden_size - 1)) == 0,
                "RMSNorm reduction requires power-of-two hidden_size");
  const std::vector<float> embeddings{
      1, 0, 0, 0,
      1, 2, 3, 4,
      0, 0, 1, 0,
  };
  const std::vector<float> update{0.5f, 0.5f, 0.5f, 0.5f};
  const std::vector<float> lm_head{
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
      1, 1, 1,
  };
  std::vector<float> logits(vocab_size);
  float *d_embeddings = nullptr, *d_hidden = nullptr, *d_update = nullptr;
  float *d_normalized = nullptr, *d_lm_head = nullptr, *d_logits = nullptr;
  CUDA_CHECK(cudaMalloc(&d_embeddings, embeddings.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hidden, hidden_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_update, hidden_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normalized, hidden_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lm_head, lm_head.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_logits, vocab_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_embeddings, embeddings.data(), embeddings.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_update, update.data(), update.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lm_head, lm_head.data(), lm_head.size() * sizeof(float), cudaMemcpyHostToDevice));

  embedding_lookup<<<1, hidden_size>>>(token_id, d_embeddings, d_hidden, hidden_size);
  residual_add<<<1, hidden_size>>>(d_update, d_hidden, hidden_size);
  rmsnorm<<<1, hidden_size, hidden_size * sizeof(float)>>>(d_hidden, d_normalized, hidden_size);
  project<<<1, vocab_size>>>(d_normalized, d_lm_head, d_logits, hidden_size, vocab_size);
  check_kernel();
  CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_embeddings)); CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_update)); CUDA_CHECK(cudaFree(d_normalized));
  CUDA_CHECK(cudaFree(d_lm_head)); CUDA_CHECK(cudaFree(d_logits));
  std::vector<float> hidden(hidden_size);
  for (int i = 0; i < hidden_size; ++i) {
    hidden[i] = embeddings[token_id * hidden_size + i] + update[i];
  }
  float sum_squares = 0.0f;
  for (float value : hidden) sum_squares += value * value;
  const float inverse_rms = 1.0f / std::sqrt(sum_squares / hidden_size + 1e-5f);
  std::vector<float> expected(vocab_size, 0.0f);
  for (int token = 0; token < vocab_size; ++token) {
    for (int i = 0; i < hidden_size; ++i) {
      expected[token] += hidden[i] * inverse_rms * lm_head[i * vocab_size + token];
    }
  }
  for (int token = 0; token < vocab_size; ++token) {
    if (!std::isfinite(logits[token]) ||
        std::fabs(logits[token] - expected[token]) > 1e-5f) {
      std::cerr << "Logit mismatch at token " << token << ": expected "
                << expected[token] << ", got " << logits[token] << '\n';
      return 1;
    }
  }
  std::cout << "LOGITS";
  for (float value : logits) std::cout << ' ' << value;
  std::cout << '\n';
  std::cout << "PASS mini_transformer_step: embedding -> residual -> RMSNorm -> logits\n";
  return 0;
}
