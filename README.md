# CUDA Kernels למתחילים על NVIDIA A100

קורס מעשי בעברית שמתחיל מ-kernel יחיד ו-thread יחיד, ומתקדם עד building blocks של LLMs כמו embeddings, ‏residual connections, ‏RMSNorm, ‏attention softmax ו-linear projections.

## למי הקורס מיועד?

לא צריך ניסיון קודם ב-CUDA. כן כדאי להכיר:

- משתנים, לולאות, פונקציות ומערכים
- C או C++ ברמה בסיסית
- הרצת פקודות ב-Linux terminal

## מה תלמדו?

בסיום תוכלו:

- להסביר מהו kernel ומה עושים `<<<blocks, threads>>>`.
- לחשב `threadIdx`, ‏`blockIdx` ואינדקס גלובלי, החל מ-0.
- להבין למה צריך bounds check.
- להעביר נתונים בין CPU ל-GPU.
- לכתוב vector addition ו-grid-stride loop.
- להבין shared memory, ‏synchronization ו-reduction.
- למפות CUDA primitives ל-embedding, ‏residual, ‏RMSNorm, ‏softmax ו-projections בתוך LLM.
- למדוד kernel באמצעות CUDA events ו-Nsight Compute.

## דרישות

- NVIDIA A100 עם driver תקין.
- CUDA Toolkit הכולל `nvcc`.
- CMake 3.24 ומעלה.
- Linux ו-compiler התומך ב-C++17.
- JupyterLab אופציונלי עבור המחברות.

ה-A100 הוא GPU מארכיטקטורת Ampere עם compute capability `8.0`, ולכן ברירת המחדל היא `sm_80`. זו מטרת compilation, לא הוכחת זהות בזמן ריצה. במכונה עם כמה GPUs בחרו A100 באמצעות `CUDA_VISIBLE_DEVICES` ואמתו עם `00_device_query`.

## התחלה מהירה

```bash
nvidia-smi
nvcc --version
export CUDA_VISIBLE_DEVICES=0  # החליפו באינדקס ה-A100 שלכם
make build
```

## מסלול 1: מתחילים מוחלטים

קראו קודם את [יסודות CUDA, צעד אחר צעד](docs/00-cuda-foundations.md), ואז הריצו כל דוגמה בנפרד:

```bash
./build/00_device_query
./build/01_hello_kernel
./build/02_one_block_index
./build/03_global_index
./build/04_bounds_check
./build/05_memory_roundtrip
./build/06_vector_add
```

או את כל מסלול הבסיס:

```bash
make run-foundations
```

אל תתקדמו לפני שאתם מסוגלים להסביר למה אינדקס 0 הוא האיבר הראשון ולמה שישה ערכים עשויים להפעיל שמונה threads.

## מסלול 2: CUDA patterns

1. [מודל הביצוע ואינדוקס](docs/01-execution-model.md)
2. [זיכרון ו-vector addition](docs/02-memory-and-vector-add.md)
3. [Grid-stride loops ו-warps](docs/03-grid-stride-and-warps.md)
4. [Shared memory ו-reduction](docs/04-shared-memory-reduction.md)
5. [כפל מטריצות ב-tiles](docs/05-tiled-matmul.md)
6. [מדידה ו-A100](docs/06-profiling-a100.md)

ה-executables המתאימים הם `06_vector_add` עד `09_tiled_matmul`.

## מסלול 3: CUDA בהקשר של LLMs

קראו את [מפת CUDA kernels בתוך LLM](docs/07-llm-kernel-map.md), ואז הריצו:

```bash
make run-llm
```

הדוגמאות:

1. `llm_01_token_embedding`: token ID בוחר embedding row.
2. `llm_02_residual_add`: חיבור residual stream.
3. `llm_03_silu_activation`: activation בתוך MLP.
4. `llm_04_rmsnorm`: reduction ו-normalization.
5. `llm_05_causal_mask`: מי רשאי לראות איזה token.
6. `llm_06_attention_softmax`: softmax יציב של attention scores.
7. `llm_07_linear_projection`: הבסיס של Q/K/V ו-LM head.
8. `llm_08_mini_transformer_step`: embedding -> residual -> RMSNorm -> logits.

אלו kernels חינוכיים. הם אינם תחליף ל-cuBLAS, ‏CUTLASS, ‏FlashAttention או kernels fused של inference engines.

## מחברות Jupyter

- [`00_indexing_cpu.ipynb`](notebooks/00_indexing_cpu.ipynb): אינדוקס ללא GPU.
- [`01_cuda_basics_a100.ipynb`](notebooks/01_cuda_basics_a100.ipynb): thread יחיד עד vector add, צעד אחר צעד.
- [`01_vector_add_a100.ipynb`](notebooks/01_vector_add_a100.ipynb): kernel מלא ראשון.
- [`02_memory_patterns_a100.ipynb`](notebooks/02_memory_patterns_a100.ipynb): grid-stride, reduction ו-tiling.
- [`03_profile_a100.ipynb`](notebooks/03_profile_a100.ipynb): Nsight Compute.
- [`04_llm_building_blocks.ipynb`](notebooks/04_llm_building_blocks.ipynb): מעבר מ-token ל-logits דרך kernels חינוכיים.

ראו [הוראות הפעלת המחברות](notebooks/README.md).

## תרגילים ובדיקות

```bash
make test
```

הבדיקות האלה הן CPU-only structural checks. הן אינן מקמפלות CUDA ואינן מחליפות הרצה על A100.

תרגילים נמצאים ב-[exercises](exercises/README.md), וכיווני פתרון ב-[solutions](solutions/README.md).

## מבנה ה-repo

```text
lessons/       יסודות CUDA ו-patterns כלליים
llm_examples/  kernels חינוכיים בהקשר של LLMs
include/       בדיקת שגיאות וטיימר GPU
docs/          הסברים בעברית
notebooks/     מחברות תרגול אינטראקטיביות
exercises/     משימות לתלמיד
solutions/     כיווני פתרון
tests/         בדיקות מבניות ללא GPU
scripts/       הרצת מסלולי הקורס
```

## כללי נכונות

- כל גישה למערך חייבת להיות בתחום התקין.
- אחרי kernel launch בודקים שגיאות. בזמן לימוד גם מסתנכרנים.
- תוצאה נכונה קודמת לאופטימיזציה.
- `256 threads/block` הוא נקודת פתיחה, לא חוק טבע.
- reduction שמחלק את מספר המשתתפים ב-2 דורש בדרך כלל block size שהוא חזקה של 2.
- softmax צריך להיות יציב נומרית: מחסרים את המקסימום לפני `exp`.

## אימות אמיתי על A100

השתמשו ב-[A100 run checklist](docs/A100-RUN-CHECKLIST.md), כולל `compute-sanitizer` ו-Nsight Compute.

## מקורות רשמיים

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)

## רישיון

MIT
