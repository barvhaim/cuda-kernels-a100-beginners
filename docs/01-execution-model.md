# שיעור 1: Grid, block, thread ו-warp

## הרעיון

Kernel הוא פונקציה שמופעלת על ה-GPU בהרבה threads. כל thread מריץ את אותו קוד, אבל מקבל מזהים שונים.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

כל האינדקסים מתחילים ב-0:

- `blockIdx.x = 2` הוא ה-block השלישי.
- `threadIdx.x = 10` הוא ה-thread ה-11 בתוך אותו block.
- אם `blockDim.x = 128`, האינדקס הגלובלי הוא `266`, כלומר האיבר ה-267.

## ההיררכיה

- Grid: כל ה-launch.
- Block: קבוצת threads שיכולה לשתף shared memory ולהסתנכרן.
- Thread: מופע לוגי אחד של ה-kernel.
- Warp: קבוצת ביצוע של 32 threads ב-NVIDIA GPU.

## למה מתחילים ב-`00_device_query`?

לפני שמניחים שיש A100, הקוד שואל את CUDA runtime מה קיים בפועל. חפשו בפלט compute capability `8.0` ו-warp size `32`.

## בדיקת הבנה

עם 4 blocks של 256 threads:

- כמה threads לוגיים הופעלו?
- איזה אינדקס גלובלי מקבל thread 7 ב-block 3?
- מהו מספרו האנושי של אותו איבר?
