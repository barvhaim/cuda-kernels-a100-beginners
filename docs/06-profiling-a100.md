# שיעור 6: מדידה ו-profiling על A100

## קודם נכונות

```bash
make run
```

כל תוכנית צריכה להדפיס `PASS`. אחר כך מודדים.

## Nsight Compute

```bash
ncu --set basic --kernel-name vector_add ./build/06_vector_add
```

או:

```bash
make profile
```

מדדים שימושיים למתחילים:

- Duration: זמן ה-kernel.
- Memory throughput: כמה מתעבורת הזיכרון הזמינה נוצלה.
- SM throughput: כמה ממשאבי החישוב נוצלו.
- Achieved occupancy: כמה warps פעילים היו ביחס למגבלה.

Occupancy גבוה אינו מטרה בפני עצמה. kernel יכול להיות memory-bound גם עם occupancy טוב.

## כללי benchmark

- בצעו warm-up לפני מדידות רציניות.
- הריצו כמה פעמים ודווחו median.
- מדדו kernel בנפרד מהעברות PCIe אם זו השאלה.
- אל תשוו ל-CPU בלי להגדיר אם זמן ההעתקות כלול.
- אל תסיקו ביצועי A100 מכך שהקוד רק עבר compilation ל-`sm_80`.

## כלי תקינות

```bash
compute-sanitizer ./build/06_vector_add
compute-sanitizer ./build/09_tiled_matmul
```

הכלי איטי יותר מהרצה רגילה, אך שימושי למציאת גישות זיכרון שגויות.
