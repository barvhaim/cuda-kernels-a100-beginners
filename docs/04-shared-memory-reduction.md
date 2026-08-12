# שיעור 4: Shared memory ו-reduction

Reduction הופך מערך לערך אחד, למשל סכום. threads באותו block טוענים ערכים ל-shared memory, מחברים בזוגות ומקטינים את מספר הערכים הפעילים בכל סיבוב.

## למה צריך synchronization?

```cpp
values[tid] = value;
__syncthreads();
```

בלי המחסום, thread עלול לקרוא תא לפני ש-thread אחר כתב אליו. `__syncthreads()` מסנכרן threads באותו block בלבד, לא את כל ה-grid.

## למה יש partial sums?

blocks שונים אינם מסתנכרנים בעזרת `__syncthreads()`. כל block כותב סכום חלקי, והדוגמה מסיימת את החיבור ב-CPU כדי לשמור על הקוד הראשון ברור.

## מלכודת

אסור שרק חלק מה-threads ב-block יגיעו ל-`__syncthreads()` כאשר אחרים מדלגים עליו. זה עלול לגרום להתנהגות לא תקינה או deadlock.
