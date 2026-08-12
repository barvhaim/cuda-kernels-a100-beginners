# מפת CUDA kernels בתוך LLM

התיקייה `llm_examples/` מחברת כל primitive של CUDA לפעולה מוכרת ב-Transformer. הדוגמאות קטנות ומכוונות ללמידה, לא לביצועי production.

כמה דוגמאות ממפות vector או attention row יחיד ל-block יחיד. לכן ה-shapes הזעירים נבחרו כך שיתאימו למספר ה-threads המרבי ב-block, ו-reductions משתמשים בגודל שהוא חזקה של 2. הגדלת shapes דורשת grid נוסף או לולאה, לא רק שינוי קבוע.

## מפת הזרימה

```text
token ID
  -> embedding lookup
  -> linear projections / attention / MLP
  -> residual add
  -> RMSNorm
  -> LM-head projection
  -> logits / softmax
```

## 1. Token embedding

קובץ: `llm_examples/01_token_embedding.cu`

Token ID הוא אינדקס של שורה בטבלת embeddings. כל block מטפל ב-token position אחד, וכל thread מעתיק hidden dimension אחד.

לומדים: אינדוקס 2D לוגי, memory layout מסוג row-major, והקשר בין token ל-hidden vector.

## 2. Residual connection

קובץ: `llm_examples/02_residual_add.cu`

```cpp
hidden_out[i] = layer_output[i] + residual[i];
```

זהו vector addition בהקשר אמיתי של LLM. שמונה threads מופעלים עבור חמישה ערכים, ולכן guard מגן על שלושת ה-threads העודפים.

## 3. SiLU activation

קובץ: `llm_examples/03_silu_activation.cu`

```text
SiLU(x) = x * sigmoid(x)
```

הדוגמה משתמשת ב-grid-stride loop. במודלי LLM רבים SiLU מופיעה ב-gated MLP, למשל כחלק מ-SwiGLU.

## 4. RMSNorm

קובץ: `llm_examples/04_rmsnorm.cu`

```text
inverse_rms = 1 / sqrt(mean(x^2) + epsilon)
output[i] = x[i] * inverse_rms * weight[i]
```

כאן מופיעים shared memory, reduction, synchronization ו-normalization. הדוגמה מטפלת ב-hidden vector אחד וב-block אחד.

## 5. Causal attention mask

קובץ: `llm_examples/05_causal_mask.cu`

כל thread מחשב תא אחד במטריצת query x key. מיקום query יכול לראות רק keys שאינם בעתיד:

```text
0 X X X
0 0 X X
0 0 0 X
0 0 0 0
```

`0` פירושו visible, ו-`X` מקבל `-INFINITY`. זוהי additive mask חינוכית ל-FP32: מחברים אותה ל-attention scores לפני softmax, ולכן ההסתברות של future position נעשית 0. ב-production צריך להתאים את ה-sentinel וה-casting ל-dtype ול-kernel בפועל.

## 6. Stable attention softmax

קובץ: `llm_examples/06_attention_softmax.cu`

הדוגמה מתחילה ב-scores סביב 1000 כדי להראות למה אסור לחשב `exp(score)` ישירות. קודם מחסרים את המקסימום:

```text
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

לומדים: max-reduction, sum-reduction ויציבות נומרית.

## 7. Linear projection

קובץ: `llm_examples/07_linear_projection.cu`

כל output dimension הוא dot product. אותו רעיון עומד בבסיס Q, K, V projections, שכבות MLP ו-LM head.

המימוש החינוכי איטי. ב-production משתמשים בדרך כלל ב-cuBLAS, ‏CUTLASS או kernels fused.

## 8. Mini transformer step

קובץ: `llm_examples/08_mini_transformer_step.cu`

הדוגמה מחברת ארבעה kernels:

```text
embedding lookup -> residual add -> RMSNorm -> logits projection
```

זה אינו Transformer מלא: אין multi-head attention, ‏KV cache או sampling. המטרה היא לראות כיצד tensors עוברים בין kernels וכיצד output של kernel אחד הופך ל-input של הבא.

## מה שונה ב-production?

- פעולות רבות fused כדי לחסוך קריאות וכתיבות ל-global memory.
- matmul משתמש ב-Tensor Cores ובפורמטים כמו BF16/FP16/FP8.
- softmax מטפל בשורות רבות, masking ויציבות נומרית מתקדמת.
- RMSNorm מטפל ב-batches וב-tokens רבים.
- inference server מנהל KV cache, batching, streams וזיכרון.

הדרך הנכונה ללמוד היא להבין קודם את הגרסה הקטנה, למדוד אותה, ואז להשוות ל-library kernel אמיתי.
