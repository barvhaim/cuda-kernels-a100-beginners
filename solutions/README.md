# כיווני פתרון

נסו קודם לבד. כאן מופיעים הכיוון והאינווריאנט החשוב, לא קוד מלא להעתקה.

## יסודות

### תרגיל 1

עם 8 threads, האינדקסים הם 0 עד 7. ‏`threadIdx.x=7` הוא ה-thread השמיני בספירה אנושית.

### תרגיל 2

```text
global = 1 * 5 + 2 = 7
```

שלושה blocks כפול חמישה threads הם 15 threads.

### תרגיל 3

```text
blocks = ceil(10 / 4) = 3
launched = 3 * 4 = 12
extra = 2
```

### תרגיל 4

ה-output הצפוי עבור `{10,20,30,40}` אחרי הכפלה הוא `{20,40,60,80}`. ראיית הבדיקה נכשלת לפני עדכון expected מוכיחה שהיא רגישה לשינוי.

### תרגיל 5

```text
blocks = ceil(1000 / 128) = 8
launched = 1024
```

האינדקסים 1000 עד 1023 חייבים להיעצר ב-guard.

## CUDA patterns

### תרגיל 6

`compute-sanitizer` אמור לזהות invalid global memory access. העובדה שהרצה רגילה לא קרסה אינה הוכחת נכונות.

### תרגיל 7

הנוסחה היא `y = alpha * x + y`. חשבו expected חדש על CPU לפני השוואת הפלט.

### תרגיל 8

אם יש N ערכים של 2, הסכום הוא `2*N`. בכל launch חדש מספר הקלטים הוא מספר ה-partials מה-launch הקודם. מספר threads נשאר חזקה של 2.

### תרגיל 9

Tile משנה block size, שימוש ב-shared memory, registers ו-occupancy. אין גודל מנצח אוניברסלי; מודדים לאחר אימות נכונות.

## LLM kernels

### תרגיל 10

Embedding table היא `[vocab_size, hidden_size]`. ‏Token ID חייב לקיים `0 <= id < vocab_size` לפני launch; אין להסתמך על kernel שיקרא מחוץ לטבלה.

### תרגיל 11

```text
blocks = ceil(13 / 8) = 2
launched = 16
extra = 3
```

### תרגיל 12

SwiGLU חינוכי משלב שני vectors:

```text
output[i] = SiLU(up[i]) * gate[i]
```

כל thread יכול לחשב איבר אחד באופן עצמאי.

### תרגיל 13

`inverse_rms` משותף לכל hidden vector; ‏`x` ו-`weight[i]` שייכים לממד מסוים. ה-reduction יוצר את הסטטיסטיקה המשותפת.

### תרגיל 14

Grid של `ceil(5/2) x ceil(5/2) = 3 x 3` blocks, עם 2x2 threads, מפעיל 36 threads עבור 25 תאים. שני ממדי ה-guard נדרשים.

### תרגיל 15

`exp(1000)` גולש ב-float. לאחר חיסור `max=1003`, הארגומנטים הם `-3,-2,-1,0`, ולכן ה-exponentials סופיים.

### תרגיל 16

כל output dimension הוא dot product אחר בין input לבין עמודה ב-weight matrix. Reference Python צריך להשתמש באותו row-major layout כמו CUDA.

### תרגיל 17

ה-shapes נשארים `[hidden_size]` דרך embedding, residual ו-RMSNorm. רק ה-LM head משנה ל-`[vocab_size]` logits.

### תרגיל 18

Fusion יכול לחסוך כתיבת intermediate ל-global memory וקריאתו מחדש. המחיר: kernel מורכב יותר, register pressure, פחות modularity ובדיקות קשות יותר.
