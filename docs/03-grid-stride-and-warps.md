# שיעור 3: Grid-stride loop ו-warps

## למה לא thread אחד לכל איבר תמיד?

אפשר להפעיל grid קטן יחסית ולתת לכל thread לטפל בכמה איברים:

```cpp
int start = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = start; i < n; i += stride) {
  y[i] = alpha * x[i] + y[i];
}
```

כך אותו kernel מתאים לגדלי קלט שונים, ומספר ה-blocks יכול להיקבע לפי מספר ה-SMs.

## Memory coalescing

threads סמוכים ב-warp ניגשים כאן לאינדקסים סמוכים. זהו pattern נוח למערכת הזיכרון. גישה כמו `x[i * 32]` בדרך כלל פחות ידידותית.

## Warp divergence

threads ב-warp מבצעים הוראות יחד. אם חצי מהם נכנסים לענף וחצי לא, שני הנתיבים עשויים להתבצע בנפרד. בדיקת גבול קצרה בקצה grid היא נורמלית; branching כבד לפי `threadIdx.x` דורש חשיבה.

## ב-A100

מספר ה-SMs משתנה בין גרסאות A100, ולכן המעבדה קוראת אותו בזמן ריצה במקום לקודד מספר קבוע.
