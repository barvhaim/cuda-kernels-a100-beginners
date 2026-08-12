# יסודות CUDA, צעד אחר צעד

המטרה בפרק הזה היא לא מהירות. המטרה היא שתוכלו להסביר כל שורת קוד לפני שמתקדמים.

## המודל המינימלי

- ה-CPU נקרא `host`.
- ה-GPU נקרא `device`.
- פונקציה עם `__global__` היא kernel שמופעל מה-CPU ורץ על ה-GPU.
- התחביר `kernel<<<blocks, threads>>>(...)` קובע כמה threads לוגיים יופעלו.
- האינדקסים מתחילים ב-0.

## 1. Kernel יחיד ו-thread יחיד

קובץ: `lessons/01_hello_kernel.cu`

```cpp
__global__ void hello_kernel() {
  printf("Hello from the GPU!\n");
}

hello_kernel<<<1, 1>>>();
```

ה-launch מכיל block אחד ובתוכו thread אחד. כאן לומדים רק את הגבול CPU -> GPU.

הרצה:

```bash
./build/01_hello_kernel
```

## 2. אינדקס בתוך block

קובץ: `lessons/02_one_block_index.cu`

```cpp
show_thread_index<<<1, 4>>>();
```

ארבעת ה-threads מקבלים `threadIdx.x` בערכים 0, 1, 2, 3. סדר שורות ה-`printf` אינו מובטח, כי threads רצים במקביל.

**קשר ל-LLM:** אפשר לתת לכל thread ממד אחד מתוך hidden state קטן.

## 3. אינדקס גלובלי

קובץ: `lessons/03_global_index.cu`

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

עם שני blocks וארבעה threads בכל block מתקבלים אינדקסים 0 עד 7:

```text
Block 0: 0 1 2 3
Block 1: 4 5 6 7
```

**קשר ל-LLM:** tensor של activations גדול מ-block אחד, ולכן צריך אינדקס ייחודי על פני כל ה-grid.

## 4. Threads עודפים ו-bounds check

קובץ: `lessons/04_bounds_check.cu`

אם יש שישה ערכים וארבעה threads בכל block:

```cpp
blocks = (6 + 4 - 1) / 4;  // 2 blocks
```

מופעלים שמונה threads. אינדקסים 6 ו-7 עודפים ולכן חייבים להיעצר ב-guard:

```cpp
if (i < n) {
  // safe access to data[i]
}
```

**קשר ל-LLM:** hidden size, מספר tokens או גודל vocabulary לא חייבים להתחלק ב-block size.

## 5. זיכרון: host ל-device ובחזרה

קובץ: `lessons/05_memory_roundtrip.cu`

המסלול הוא:

```text
CPU vector
  -> cudaMalloc
  -> cudaMemcpy HostToDevice
  -> kernel adds 1
  -> cudaMemcpy DeviceToHost
  -> CPU verifies result
  -> cudaFree
```

**קשר ל-LLM:** token IDs, weights ו-activations חייבים להיות בזיכרון שנגיש ל-GPU בזמן kernel execution.

## 6. Vector addition

קובץ: `lessons/06_vector_add.cu`

כעת מחברים אינדוקס, guard וזיכרון לתוכנית מלאה. כל thread מחשב איבר אחד:

```cpp
c[i] = a[i] + b[i];
```

**קשר ל-LLM:** זהו אותו pattern של residual connection:

```text
hidden = layer_output + residual
```

## לפני שממשיכים

אתם מוכנים ל-grid-stride loop רק אם אתם יכולים לענות בלי להסתכל:

1. למה `threadIdx.x = 10` הוא ה-thread ה-11?
2. למה `blockIdx.x * blockDim.x` מופיע באינדקס הגלובלי?
3. למה מפעילים לעיתים יותר threads ממספר הערכים?
4. מה ההבדל בין `host_values` ל-`device_values`?
5. למה kernel launch זקוק לבדיקת שגיאות ול-synchronization בזמן לימוד?
