# CUDA Kernels for Beginners on NVIDIA A100

A practical English course that starts with one kernel and one thread, then progresses to LLM building blocks such as embeddings, residual connections, RMSNorm, attention softmax, and linear projections.

## Who is this course for?

No previous CUDA experience is required. You should be comfortable with:

- Variables, loops, functions, and arrays
- Basic C or C++
- Running commands in a Linux terminal

## What will you learn?

By the end of the course, you will be able to:

- Explain what a kernel is and what `<<<blocks, threads>>>` means.
- Compute `threadIdx`, `blockIdx`, and a zero-based global index.
- Explain why CUDA kernels need bounds checks.
- Move data between the CPU and GPU.
- Write vector addition and grid-stride loops.
- Understand shared memory, synchronization, and reduction.
- Map CUDA primitives to embeddings, residual connections, RMSNorm, softmax, and projections in an LLM.
- Measure kernels with CUDA events and Nsight Compute.

## Requirements

- An NVIDIA A100 with a working driver
- CUDA Toolkit, including `nvcc`
- CMake 3.24 or newer
- Linux and a compiler with C++17 support
- JupyterLab, optionally, for the notebooks

The A100 is an Ampere GPU with compute capability `8.0`, so the default build target is `sm_80`. This is a compilation target, not proof of the runtime GPU identity. On a multi-GPU machine, select an A100 with `CUDA_VISIBLE_DEVICES` and verify it with `00_device_query`.

## Quick start

```bash
nvidia-smi
nvcc --version
export CUDA_VISIBLE_DEVICES=0  # Replace with the index of your A100
make build
```

## Track 1: Complete beginners

Start with [CUDA Foundations, Step by Step](docs/00-cuda-foundations.md), then run each example separately:

```bash
./build/00_device_query
./build/01_hello_kernel
./build/02_one_block_index
./build/03_global_index
./build/04_bounds_check
./build/05_memory_roundtrip
./build/06_vector_add
```

Or run the complete foundations track:

```bash
make run-foundations
```

Do not move on until you can explain why index 0 is the first element and why six values may launch eight threads.

## Track 2: CUDA patterns

1. [Execution Model and Indexing](docs/01-execution-model.md)
2. [Memory and Vector Addition](docs/02-memory-and-vector-add.md)
3. [Grid-Stride Loops and Warps](docs/03-grid-stride-and-warps.md)
4. [Shared Memory and Reduction](docs/04-shared-memory-reduction.md)
5. [Tiled Matrix Multiplication](docs/05-tiled-matmul.md)
6. [Profiling on A100](docs/06-profiling-a100.md)

The corresponding executables are `06_vector_add` through `09_tiled_matmul`.

## Track 3: CUDA in LLMs

Read [A Map of CUDA Kernels Inside an LLM](docs/07-llm-kernel-map.md), then run:

```bash
make run-llm
```

The examples are:

1. `llm_01_token_embedding`: a token ID selects an embedding row.
2. `llm_02_residual_add`: add the residual stream.
3. `llm_03_silu_activation`: an activation used inside an MLP.
4. `llm_04_rmsnorm`: reduction and normalization.
5. `llm_05_causal_mask`: control which tokens may be attended to.
6. `llm_06_attention_softmax`: numerically stable softmax over attention scores.
7. `llm_07_linear_projection`: the foundation of Q/K/V projections and the LM head.
8. `llm_08_mini_transformer_step`: embedding -> residual -> RMSNorm -> logits.

These are educational kernels. They do not replace cuBLAS, CUTLASS, FlashAttention, or fused kernels in production inference engines.

## Jupyter notebooks

- [`00_indexing_cpu.ipynb`](notebooks/00_indexing_cpu.ipynb): learn indexing without a GPU.
- [`01_cuda_basics_a100.ipynb`](notebooks/01_cuda_basics_a100.ipynb): one thread through vector addition, step by step.
- [`01_vector_add_a100.ipynb`](notebooks/01_vector_add_a100.ipynb): your first complete CUDA program.
- [`02_memory_patterns_a100.ipynb`](notebooks/02_memory_patterns_a100.ipynb): grid-stride loops, reduction, and tiling.
- [`03_profile_a100.ipynb`](notebooks/03_profile_a100.ipynb): profile with Nsight Compute.
- [`04_llm_building_blocks.ipynb`](notebooks/04_llm_building_blocks.ipynb): follow a token through educational kernels to logits.

See the [notebook instructions](notebooks/README.md).

## Exercises and tests

```bash
make test
```

These tests are CPU-only structural checks. They do not compile CUDA and do not replace validation on an A100.

Exercises are in [exercises](exercises/README.md), with solution guidance in [solutions](solutions/README.md).

## Repository structure

```text
lessons/       CUDA foundations and general patterns
llm_examples/  educational kernels in an LLM context
include/       CUDA error checking and a GPU timer
docs/          course explanations in English
notebooks/     interactive practice notebooks
exercises/     student tasks
solutions/     solution guidance
tests/         structural checks that do not require a GPU
scripts/       course-track runners
```

## Correctness rules

- Every array access must stay within the valid range.
- Check errors after every kernel launch. While learning, synchronize as well.
- Correctness comes before optimization.
- `256 threads/block` is a starting point, not a law.
- A halving reduction usually requires a power-of-two block size.
- Softmax must be numerically stable: subtract the maximum before `exp`.

## Real validation on A100

Follow the [A100 run checklist](docs/A100-RUN-CHECKLIST.md), including `compute-sanitizer` and Nsight Compute.

## Official references

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)

## License

MIT
