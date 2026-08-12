# CUDA Kernels למתחילים על NVIDIA A100

קורס מעשי וקצר שמלמד כתיבת CUDA kernels מהאינדקס הראשון ועד shared memory וכפל מטריצות.

## מה בונים

בסוף המסלול תוכלו:

- להסביר `grid`, `block`, `thread` ו-`warp`.
- לחשב אינדקס גלובלי בלי להתבלבל בין אינדקס למיקום אנושי.
- להקצות ולהעתיק זיכרון בין host ל-device.
- לכתוב kernel עם בדיקת גבולות ו-grid-stride loop.
- להשתמש ב-`shared memory` וב-`__syncthreads()`.
- למדוד זמן GPU ולבצע profiling בסיסי ב-Nsight Compute.
- להבין מה עדיין חסר בדרך ל-kernels מהירים של LLMs.

## דרישות

- כרטיס NVIDIA A100 עם driver תקין.
- CUDA Toolkit הכולל `nvcc`.
- CMake 3.24 ומעלה.
- Linux ו-compiler התומך ב-C++17.
- אין צורך ב-Python packages לבדיקות ה-repo.

ה-A100 הוא GPU מארכיטקטורת Ampere עם compute capability `8.0`, ולכן ברירת המחדל היא `sm_80`. זו מטרת compilation, לא בדיקת זהות בזמן ריצה: במכונה עם כמה GPUs בחרו A100 באמצעות `CUDA_VISIBLE_DEVICES` ואמתו עם `00_device_query`.

## התחלה מהירה, 10-15 דקות

```bash
nvidia-smi
nvcc --version
export CUDA_VISIBLE_DEVICES=0  # בחרו כאן את אינדקס ה-A100 מהמכונה
make build
./build/00_device_query
./build/01_vector_add
```

הצלחה נראית בקירוב כך, כשהמספרים תלויים במכונה:

```text
Device 0: NVIDIA A100 ...
  compute capability: 8.0
  warp size: 32
PASS vector_add: 1048576 values in ... ms
```

להרצת כל המעבדות:

```bash
make run
```

לבדיקות מבניות, גם במכונה ללא CUDA:

```bash
make test
```

## מחברות Jupyter

המחברות משלבות הסבר, קוד, שאלות וניסויים:

1. [`00_indexing_cpu.ipynb`](notebooks/00_indexing_cpu.ipynb): אינדוקס ובדיקות גבול ללא GPU.
2. [`01_vector_add_a100.ipynb`](notebooks/01_vector_add_a100.ipynb): בנייה והרצת kernel ראשון.
3. [`02_memory_patterns_a100.ipynb`](notebooks/02_memory_patterns_a100.ipynb): grid-stride, reduction ו-tiling.
4. [`03_profile_a100.ipynb`](notebooks/03_profile_a100.ipynb): profiling עם Nsight Compute.

הוראות הפעלה נמצאות ב-[מדריך המחברות](notebooks/README.md).

## מסלול הלימוד

1. [מודל הביצוע ואינדוקס](docs/01-execution-model.md), ואז `00_device_query`.
2. [זיכרון ו-vector addition](docs/02-memory-and-vector-add.md), ואז `01_vector_add`.
3. [Grid-stride loops ו-warps](docs/03-grid-stride-and-warps.md), ואז `02_grid_stride`.
4. [Shared memory ו-reduction](docs/04-shared-memory-reduction.md), ואז `03_reduction`.
5. [כפל מטריצות ב-tiles](docs/05-tiled-matmul.md), ואז `04_tiled_matmul`.
6. [מדידה ו-A100](docs/06-profiling-a100.md).
7. [תרגילים](exercises/README.md), ורק אחר כך [פתרונות](solutions/README.md).

## מבנה ה-repo

```text
lessons/      קוד CUDA מלא וניתן להרצה
include/      בדיקת שגיאות וטיימר GPU
docs/         הסברים בעברית
notebooks/    מחברות תרגול אינטראקטיביות
exercises/    משימות לשינוי הקוד
solutions/    כיווני פתרון, לא העתק מלא של כל מעבדה
tests/        בדיקות תקינות מבניות ללא GPU
scripts/      הרצת כל המעבדות
```

## כללי בטיחות ונכונות

- כל גישה למערך חייבת להיות בתחום התקין.
- אחרי launch יש לבדוק שגיאות. בזמן לימוד משתמשים גם ב-`cudaDeviceSynchronize()`.
- אין להשוות זמני kernel באמצעות שעון CPU בלי synchronization.
- תוצאה נכונה קודמת לאופטימיזציה.
- `256 threads/block` הוא נקודת פתיחה סבירה, לא חוק טבע.

## גבולות הקורס

הקורס מכוון ליסודות. ה-matmul כאן מדגים tiling אבל אינו מתחרה ב-cuBLAS, CUTLASS או Tensor Cores. בהמשך אפשר להוסיף streams, pinned memory, warp primitives, mixed precision ו-custom PyTorch extensions.

## מקורות רשמיים

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA A100 Tensor Core GPU Architecture](https://www.nvidia.com/en-us/data-center/a100/)

## רישיון

MIT
