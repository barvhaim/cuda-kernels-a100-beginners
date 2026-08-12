# מחברות הקורס

מומלץ להריץ Jupyter משורש ה-repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab
jupyter lab
```

הסדר המומלץ:

1. `00_indexing_cpu.ipynb`: עובד גם בלי GPU.
2. `01_vector_add_a100.ipynb`: kernel ראשון ו-build אמיתי.
3. `02_memory_patterns_a100.ipynb`: שלושה patterns מרכזיים.
4. `03_profile_a100.ipynb`: Nsight Compute.

המחברות 1-3 מניחות A100, ‏CUDA Toolkit ו-build tools. הן נכשלות במפורש כאשר prerequisite חסר, במקום להציג פלט מדומה.
