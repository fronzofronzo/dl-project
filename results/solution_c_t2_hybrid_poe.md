# Solution C — Φ-Flow = t2_hybrid_poe

PoE re-rank: η=0.03, weighting=acc, composition: t2 (train-tuned η, single test run)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.186 | 0.051 | 0.260 | 0.041 | 4786 |
| +Eyeglasses | 0.115 | 0.115 | 0.383 | 0.101 | 0.569 | 0.097 | 2196 |
| -Heavy_Makeup | 0.049 | 0.049 | 0.148 | 0.035 | 0.231 | 0.030 | 4087 |
| +Male | 0.087 | 0.087 | 0.255 | 0.074 | 0.354 | 0.070 | 1595 |
| -Young | 0.042 | 0.042 | 0.145 | 0.037 | 0.231 | 0.034 | 5355 |
| +Blond_Hair | 0.063 | 0.063 | 0.222 | 0.058 | 0.333 | 0.053 | 5469 |
| +Mustache | 0.123 | 0.123 | 0.332 | 0.087 | 0.439 | 0.070 | 301 |
| +Eyeglasses, +Smiling | 0.181 | 0.181 | 0.405 | 0.115 | 0.536 | 0.093 | 612 |
| +Black_Hair, -Wavy_Hair | 0.056 | 0.056 | 0.184 | 0.045 | 0.297 | 0.043 | 2572 |
| -Male, -Mustache | 0.074 | 0.074 | 0.185 | 0.044 | 0.259 | 0.033 | 27 |
| +Chubby, -Young | 0.086 | 0.086 | 0.296 | 0.064 | 0.442 | 0.056 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.228 | 0.646 | 0.220 | 0.886 | 0.190 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.059 | 0.012 | 0.059 | 0.006 | 34 |
| MACRO | 0.090 | 0.090 | 0.265 | 0.073 | 0.377 | 0.063 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t2_hybrid_poe R@1 | naive R@5 | t2_hybrid_poe R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.070 | 0.183 | 0.186 |
| +Eyeglasses | 0.018 | 0.115 | 0.077 | 0.383 |
| -Heavy_Makeup | 0.029 | 0.049 | 0.094 | 0.148 |
| +Male | 0.006 | 0.087 | 0.012 | 0.255 |
| -Young | 0.009 | 0.042 | 0.031 | 0.145 |
| +Blond_Hair | 0.024 | 0.063 | 0.082 | 0.222 |
| +Mustache | 0.063 | 0.123 | 0.169 | 0.332 |
| +Eyeglasses, +Smiling | 0.018 | 0.181 | 0.051 | 0.405 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.056 | 0.098 | 0.184 |
| -Male, -Mustache | 0.000 | 0.074 | 0.000 | 0.185 |
| +Chubby, -Young | 0.003 | 0.086 | 0.012 | 0.296 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.228 | 0.063 | 0.646 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.059 |
| MACRO | 0.022 | 0.090 | 0.072 | 0.265 |

## vs t2-hybrid (R@1 / R@5)

| query | t2-hybrid R@1 | t2_hybrid_poe R@1 | t2-hybrid R@5 | t2_hybrid_poe R@5 |
|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.191 | 0.186 |
| +Eyeglasses | 0.118 | 0.115 | 0.383 | 0.383 |
| -Heavy_Makeup | 0.040 | 0.049 | 0.133 | 0.148 |
| +Male | 0.087 | 0.087 | 0.255 | 0.255 |
| -Young | 0.040 | 0.042 | 0.144 | 0.145 |
| +Blond_Hair | 0.066 | 0.063 | 0.221 | 0.222 |
| +Mustache | 0.116 | 0.123 | 0.306 | 0.332 |
| +Eyeglasses, +Smiling | 0.131 | 0.181 | 0.342 | 0.405 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.056 | 0.171 | 0.184 |
| -Male, -Mustache | 0.074 | 0.074 | 0.148 | 0.185 |
| +Chubby, -Young | 0.149 | 0.086 | 0.363 | 0.296 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.228 | 0.468 | 0.646 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.059 |
| MACRO | 0.090 | 0.090 | 0.240 | 0.265 |
