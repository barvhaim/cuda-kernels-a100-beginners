# שיעור 2: זיכרון וחיבור וקטורים

## המסלול

```text
host vectors -> cudaMemcpy HostToDevice -> kernel -> cudaMemcpy DeviceToHost -> verification
```

`cudaMalloc` מקצה device memory. ה-kernel מקבל `d_a`, `d_b`, `d_c`, ולא את ה-vectors של ה-CPU.

## ה-kernel

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
```

מספר ה-blocks מחושב בעיגול כלפי מעלה:

```cpp
int blocks = (n + threads - 1) / threads;
```

לכן ייתכן שיופעלו threads עודפים. התנאי `i < n` מונע גישה מחוץ למערך.

## אסינכרוניות

Kernel launch הוא בדרך כלל אסינכרוני ביחס ל-CPU. `CudaTimer` משתמש ב-CUDA events, והעתקה סינכרונית חזרה ל-host ממתינה לעבודה הדרושה.

## משימת קריאה

פתחו `lessons/06_vector_add.cu` וסמנו את ששת השלבים: host allocation, device allocation, H2D, launch, D2H, verification/free.
