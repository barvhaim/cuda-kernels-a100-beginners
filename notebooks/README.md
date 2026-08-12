# מחברות הקורס

מומלץ להריץ Jupyter משורש ה-repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab
jupyter lab
```

## סדר מומלץ

1. `00_indexing_cpu.ipynb`: אינדוקס ובדיקת גבולות ללא GPU.
2. `01_cuda_basics_a100.ipynb`: thread יחיד, אינדוקס, bounds, memory ו-vector add.
3. `01_vector_add_a100.ipynb`: העמקה בתוכנית CUDA מלאה.
4. `02_memory_patterns_a100.ipynb`: grid-stride, reduction ו-tiling.
5. `04_llm_building_blocks.ipynb`: embeddings, residual, RMSNorm, mask, softmax ו-projection.
6. `03_profile_a100.ipynb`: profiling רק אחרי שמבינים ונבדקה נכונות.

מחברות A100 מחפשות את שורש ה-repo כלפי מעלה. הן בודקות prerequisites ונכשלות במפורש כאשר CUDA או כלי נדרש חסרים, במקום להציג פלט מדומה.

מחברת LLM כוללת תחילה reference קטן ב-Python. אפשר להריץ את החלק הזה גם ללא GPU; תאי ה-build וההרצה דורשים A100.
