# Lesson 6: Measurement and Profiling on A100

## Correctness first

```bash
make run
```

Every program should print `PASS`. Measure only after correctness is established.

## Nsight Compute

```bash
ncu --set basic --kernel-name vector_add ./build/06_vector_add
```

Or run:

```bash
make profile
```

Useful beginner metrics include:

- Duration: kernel execution time.
- Memory throughput: the fraction of available memory bandwidth used.
- SM throughput: the fraction of compute resources used.
- Achieved occupancy: active warps relative to the hardware limit.

High occupancy is not a goal by itself. A kernel can remain memory-bound even with good occupancy.

## Benchmark rules

- Warm up before serious measurements.
- Run several times and report the median.
- Measure the kernel separately from PCIe transfers when that is the question.
- Do not compare with a CPU result without stating whether transfer time is included.
- Do not infer A100 performance merely because the code compiled for `sm_80`.

## Correctness tools

```bash
compute-sanitizer ./build/06_vector_add
compute-sanitizer ./build/09_tiled_matmul
```

The sanitizer is slower than a normal run, but it is useful for finding invalid memory accesses.
