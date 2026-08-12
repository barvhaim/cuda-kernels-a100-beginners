# A100 run checklist

המסמך הזה מפריד בין תקינות ה-repo לבין אימות אמיתי על GPU.

## סביבה

```bash
nvidia-smi
nvcc --version
cmake --version
```

במכונה עם כמה GPUs, מצאו את אינדקס ה-A100 ובחרו אותו לפני ההרצה:

```bash
export CUDA_VISIBLE_DEVICES=<A100-index>
```

ודאו שה-device query מדווח שם המכיל `A100` ו-compute capability `8.0`. ‏`sm_80` לבדו אינו מוכיח שהחומרה היא A100.

## Build והרצה

```bash
make clean
make build
make run-foundations
make run-patterns
make run-llm
```

נדרש `PASS` מכל 18 ה-executables: עשרה שיעורי CUDA כלליים ושמונה kernels חינוכיים של LLM.

## בדיקת זיכרון

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

נדרש: אפס שגיאות.

## Profiling בסיסי

```bash
ncu --set basic --kernel-name vector_add ./build/06_vector_add
ncu --set basic --kernel-name tiled_matmul ./build/09_tiled_matmul
```

שמרו את דגם ה-A100 המדויק, גרסת driver, גרסת CUDA, זמן kernel ו-metrics מרכזיים. אל תכניסו מספרי ביצועים ל-README לפני שהפקודות האלה הורצו בפועל.
