# תרגילים

עבדו לפי הסדר. בכל תרגיל: נחשו קודם, שנו דבר אחד, הריצו, ורשמו מה קרה.

## חלק א: יסודות

### תרגיל 1: מתחילים מ-0

ב-`02_one_block_index`, הפעילו 8 threads. כתבו לפני ההרצה מהם האינדקס הראשון והאחרון ומהו מספרו האנושי של `threadIdx.x=7`.

### תרגיל 2: אינדקס גלובלי

ב-`03_global_index`, הפעילו 3 blocks עם 5 threads בכל block. חשבו ידנית את האינדקס של thread 2 ב-block 1 ואת מספר ה-threads הכולל.

### תרגיל 3: Threads עודפים

ב-`04_bounds_check`, שנו ל-`n=10` ו-`threads=4`. נחשו כמה blocks, כמה threads וכמה threads עודפים יהיו.

### תרגיל 4: Host ו-device

ב-`05_memory_roundtrip`, שנו את ה-kernel מ-`+1` ל-`*2`. עדכנו את expected output רק לאחר שראיתם את הבדיקה נכשלת.

### תרגיל 5: Vector addition

שנו את `06_vector_add` ל-1000 איברים ו-128 threads/block. הדפיסו את מספר ה-blocks והסבירו למה מופעלים יותר מ-1000 threads.

## חלק ב: CUDA patterns

### תרגיל 6: בכוונה בלי guard

צרו עותק מקומי של vector add, הסירו את `if (i < n)` והריצו רק דרך `compute-sanitizer`. אל תשמרו את הגרסה השבורה. תעדו מה הכלי מצא.

### תרגיל 7: Grid-stride SAXPY

שנו את `alpha` ואת ערכי הקלט ב-`07_grid_stride`. עדכנו את expected value וודאו שהבדיקה נכשלת לפני העדכון ועוברת אחריו.

### תרגיל 8: Reduction

שנו את הקלט ב-`08_reduction` לערכים `2.0f`. ה-kernel דורש `threads` שהוא חזקה של 2. אל תשנו ל-96 או ל-192. אחר כך החליפו את סיום ה-CPU ב-launches חוזרים עם `threads=256` עד שנשאר סכום יחיד.

### תרגיל 9: Matmul

נסו `TILE=8`, `16`, `32` ב-`09_tiled_matmul`. ודאו נכונות ומדדו. אין להניח שה-tile הגדול ביותר יהיה המהיר ביותר.

## חלק ג: LLM kernels

### תרגיל 10: Embedding lookup

ב-`llm_01_token_embedding`, הוסיפו token חמישי לטבלה ושנו את ה-token IDs. ציירו את shape של הטבלה ושל output. לפני שימוש ב-token ID שאינו בתחום, הוסיפו validation בצד ה-CPU שמסרב לבצע את ה-kernel launch. ודאו שהקלט השגוי נדחה בבטחה.

### תרגיל 11: Residual connection

שנו את hidden vector לאורך 13 ואת block size ל-8. חשבו כמה threads עודפים יהיו והסבירו למה guard נחוץ.

### תרגיל 12: SiLU ו-SwiGLU

הריצו SiLU על 9 ערכים עם 4 threads. לאחר מכן הוסיפו vector שני `gate` וחשבו גרסה חינוכית:

```text
output[i] = SiLU(up[i]) * gate[i]
```

### תרגיל 13: RMSNorm

שנו את weights כך שלא יהיו כולם 1. השוו ל-reference ב-Python. הסבירו אילו ערכים משותפים לכל ה-threads ואילו פרטיים לכל thread.

### תרגיל 14: Causal mask

שנו את sequence length ל-5. ודאו ש-grid של 2x2 threads עדיין מכסה את כל 25 התאים בעזרת bounds checks.

### תרגיל 15: Stable softmax

הסירו זמנית את חיסור המקסימום והשאירו scores סביב 1000. תעדו את `inf` או `nan`, החזירו את התיקון והריצו שוב.

### תרגיל 16: Q projection

שנו את `llm_07_linear_projection` מ-output size של 2 ל-3. כתבו reference ב-Python והשוו כל output dimension.

### תרגיל 17: Mini transformer step

הדפיסו או העתיקו חזרה ל-host גם את hidden state אחרי embedding, אחרי residual ואחרי RMSNorm. תעדו את shape והערכים בכל boundary.

### תרגיל 18: מה כדאי לבצע ב-fusion?

בחרו שתי פעולות סמוכות ב-mini pipeline. הסבירו אילו כתיבות וקריאות ל-global memory אפשר לחסוך אם מאחדים אותן, ומה המחיר מבחינת מורכבות ובדיקות.

## ניסוי A100

הריצו Nsight Compute על `06_vector_add`, ‏`llm_04_rmsnorm` ו-`llm_07_linear_projection`. לכל kernel כתבו השערה: memory-bound או compute-bound, והביאו שני metrics שתומכים בה.

## יומן התקדמות

לכל תרגיל רשמו:

- הפקודה שהרצתם.
- shape של כל input ו-output.
- התוצאה הצפויה והאמיתית.
- כשל אחד שראיתם ומה גרם לו.
- דבר אחד שאתם מסוגלים להסביר עכשיו בלי להסתכל בקוד.
