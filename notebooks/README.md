# Course Notebooks

Run Jupyter from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab
jupyter lab
```

## Recommended order

1. `00_indexing_cpu.ipynb`: indexing and bounds checks without a GPU.
2. `01_cuda_basics_a100.ipynb`: one thread, indexing, bounds, memory, and vector addition.
3. `01_vector_add_a100.ipynb`: a deeper look at one complete CUDA program.
4. `02_memory_patterns_a100.ipynb`: grid-stride loops, reduction, and tiling.
5. `04_llm_building_blocks.ipynb`: embeddings, residual addition, RMSNorm, masking, softmax, and projection.
6. `03_profile_a100.ipynb`: profile only after you understand the code and verify correctness.

The A100 notebooks search upward for the repository root. They check prerequisites and fail explicitly when CUDA or another required tool is missing instead of displaying fabricated output.

The LLM notebook starts with a small Python reference. You can run that section without a GPU; the build and execution cells require an A100.
