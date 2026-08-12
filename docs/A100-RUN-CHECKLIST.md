# A100 Run Checklist

This checklist separates repository consistency from real GPU validation.

## Environment

```bash
nvidia-smi
nvcc --version
cmake --version
```

On a multi-GPU machine, identify the A100 index and select it before running the course:

```bash
export CUDA_VISIBLE_DEVICES=<A100-index>
```

Verify that the device query reports a name containing `A100` and compute capability `8.0`. An `sm_80` build target alone does not prove that the runtime GPU is an A100.

## Build and run

```bash
make clean
make build
make run-foundations
make run-patterns
make run-llm
```

All 18 executables should report `PASS`: ten general CUDA lessons and eight educational LLM kernels.

## Memory validation

```bash
compute-sanitizer ./build/06_vector_add
compute-sanitizer ./build/07_grid_stride
compute-sanitizer ./build/08_reduction
compute-sanitizer ./build/09_tiled_matmul
compute-sanitizer ./build/llm_01_token_embedding
compute-sanitizer ./build/llm_04_rmsnorm
compute-sanitizer ./build/llm_05_causal_mask
compute-sanitizer ./build/llm_06_attention_softmax
compute-sanitizer ./build/llm_08_mini_transformer_step
```

Required result: zero errors.

## Basic profiling

```bash
ncu --set basic --kernel-name vector_add ./build/06_vector_add
ncu --set basic --kernel-name tiled_matmul ./build/09_tiled_matmul
```

Record the exact A100 model, driver version, CUDA version, kernel time, and important metrics. Do not add performance numbers to the README before running these commands on the real hardware.
