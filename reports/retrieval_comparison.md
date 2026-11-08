# Chunking comparison

Top-k = 3 chunks; each result is mapped to its parent document and deduplicated before scoring.

## fixed

| Query | Relevant | Retrieved documents | Precision | Recall |
|---|---|---|---|---|
| Q01 | returns | returns, warranty | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q02 | cod_refunds | cancellation, cod_refunds | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q03 | delivery | delivery, international | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q04 | reverse_pickup | loyalty, reverse_pickup | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q05 | warranty | returns, warranty | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q06 | cancellation | cancellation, delivery | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q07 | loyalty | cod_refunds, loyalty | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q08 | payment_retry | cod_refunds, payment_retry | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q09 | size_exchange | returns, size_exchange | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q10 | damaged_claim | damaged_claim, payment_retry | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q11 | international | delivery, international | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q12 | escalation | delivery, escalation | 1/2 = 0.500000 | 1/1 = 1.000000 |

Macro precision = 0.500000; macro recall = 1.000000; macro F1 = 0.666667.

## sentence

| Query | Relevant | Retrieved documents | Precision | Recall |
|---|---|---|---|---|
| Q01 | returns | returns, warranty | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q02 | cod_refunds | cod_refunds, payment_retry | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q03 | delivery | delivery, international | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q04 | reverse_pickup | cod_refunds, reverse_pickup | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q05 | warranty | returns, warranty | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q06 | cancellation | cancellation, delivery | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q07 | loyalty | escalation, loyalty | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q08 | payment_retry | cod_refunds, payment_retry | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q09 | size_exchange | returns, size_exchange | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q10 | damaged_claim | damaged_claim, warranty | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q11 | international | delivery, international | 1/2 = 0.500000 | 1/1 = 1.000000 |
| Q12 | escalation | delivery, escalation | 1/2 = 0.500000 | 1/1 = 1.000000 |

Macro precision = 0.500000; macro recall = 1.000000; macro F1 = 0.666667.

Deploy **sentence**, selected by macro F1 on these 12 queries (ties prefer complete sentence boundaries).
This is a small development set; the same queries are also used for demonstrations, so these figures are descriptive rather than an unbiased generalization estimate.
