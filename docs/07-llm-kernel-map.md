# A Map of CUDA Kernels Inside an LLM

The `llm_examples/` directory connects CUDA primitives to familiar Transformer operations. These examples are intentionally small and educational, not production implementations.

Several examples map one vector or one attention row to a single block. Their tiny shapes fit the maximum number of threads in a block, and their reductions use power-of-two sizes. Scaling these examples requires another grid dimension or a loop, not merely changing a constant.

## The data flow

```text
token ID
  -> embedding lookup
  -> linear projections / attention / MLP
  -> residual add
  -> RMSNorm
  -> LM-head projection
  -> logits / softmax
```

## 1. Token embedding

File: `llm_examples/01_token_embedding.cu`

A token ID selects a row from the embedding table. Each block handles one token position, and each thread copies one hidden dimension.

You learn logical two-dimensional indexing, row-major memory layout, and the relationship between a token and its hidden vector. Both the host and kernel validate token IDs before accessing the table.

## 2. Residual connection

File: `llm_examples/02_residual_add.cu`

```cpp
hidden_out[i] = layer_output[i] + residual[i];
```

This is vector addition in a real LLM context. Eight threads are launched for five values, so a bounds check protects the three extra threads.

## 3. SiLU activation

File: `llm_examples/03_silu_activation.cu`

```text
SiLU(x) = x * sigmoid(x)
```

The example uses a grid-stride loop and a numerically stable sigmoid formula, including a test input of `-100`. Many LLMs use SiLU in a gated MLP, for example as part of SwiGLU.

## 4. RMSNorm

File: `llm_examples/04_rmsnorm.cu`

```text
inverse_rms = 1 / sqrt(mean(x^2) + epsilon)
output[i] = x[i] * inverse_rms * weight[i]
```

This example combines shared memory, reduction, synchronization, and normalization. It handles one hidden vector with one block and explicitly enforces that one-block limitation.

## 5. Causal attention mask

File: `llm_examples/05_causal_mask.cu`

Each thread computes one cell in a query-by-key matrix. A query position can see only keys that are not in the future:

```text
0 X X X
0 0 X X
0 0 0 X
0 0 0 0
```

`0` means visible, and `X` receives `-INFINITY`. This is an educational additive mask for FP32: add it to the attention scores before softmax, and future positions receive probability 0. In production, choose the sentinel and casting behavior according to the actual dtype and fused kernel.

## 6. Stable attention softmax

File: `llm_examples/06_attention_softmax.cu`

The example starts with scores near 1000 to show why computing `exp(score)` directly is unsafe. Subtract the maximum first:

```text
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

You learn max reduction, sum reduction, and numerical stability. The program compares every probability with a CPU reference and rejects non-finite or negative outputs.

## 7. Linear projection

File: `llm_examples/07_linear_projection.cu`

Each output dimension is a dot product. The same idea underlies Q, K, and V projections, MLP layers, and the LM head.

The educational implementation is slow. Production code normally uses cuBLAS, CUTLASS, or fused kernels.

## 8. Mini Transformer step

File: `llm_examples/08_mini_transformer_step.cu`

This example connects four kernels:

```text
embedding lookup -> residual add -> RMSNorm -> logits projection
```

It is not a complete Transformer: it has no multi-head attention, KV cache, or sampling. Its purpose is to show how tensors move between kernels and how one kernel's output becomes the next kernel's input. The final logits are compared with a CPU reference.

## What changes in production?

- Many operations are fused to avoid repeated global-memory reads and writes.
- Matrix multiplication uses Tensor Cores and formats such as BF16, FP16, or FP8.
- Softmax handles many rows, masking, and more advanced numerical concerns.
- RMSNorm handles batches and many token positions.
- An inference server manages the KV cache, batching, streams, and memory.

The right learning path is to understand the small version first, measure it, and then compare it with a real library kernel.
