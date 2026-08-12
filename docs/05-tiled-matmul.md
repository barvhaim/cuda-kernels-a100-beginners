# שיעור 5: Tiled matrix multiplication

בכפל מטריצות נאיבי, ערכים של A ושל B נקראים שוב ושוב מ-global memory. Tiling טוען חתיכות קטנות ל-shared memory ומשתמש בהן מחדש.

## מיפוי דו-ממדי

```cpp
int row = blockIdx.y * TILE + threadIdx.y;
int col = blockIdx.x * TILE + threadIdx.x;
```

כל block מחשב tile של C. בכל סבב הוא טוען tile מ-A ו-tile מ-B, מסתנכרן, מבצע מכפלות, ומסתנכרן לפני הטעינה הבאה.

## למה N הוא 257?

כדי לבדוק שהקוד אינו מניח שהממד מתחלק ב-16. טעינות מחוץ לטווח מוחלפות ב-0, וכתיבת הפלט מוגנת בבדיקת גבול.

## מה הדוגמה אינה עושה

היא אינה משתמשת ב-Tensor Cores ואינה kernel תחרותי. על A100, matmul אמיתי צריך בדרך כלל cuBLAS או CUTLASS. מטרת הקוד היא להבין reuse, synchronization ואינדוקס 2D.
