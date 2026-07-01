# Solution A — T2 learned directions (Φ = t2_hybrid_tau05_fixed)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.184 | 0.050 | 0.259 | 0.041 | 4786 |
| +Eyeglasses | 0.119 | 0.119 | 0.395 | 0.105 | 0.560 | 0.096 | 2196 |
| -Heavy_Makeup | 0.037 | 0.037 | 0.118 | 0.027 | 0.188 | 0.024 | 4087 |
| +Male | 0.070 | 0.070 | 0.213 | 0.059 | 0.310 | 0.055 | 1595 |
| -Young | 0.041 | 0.041 | 0.138 | 0.035 | 0.213 | 0.031 | 5355 |
| +Blond_Hair | 0.067 | 0.067 | 0.213 | 0.055 | 0.329 | 0.050 | 5469 |
| +Mustache | 0.159 | 0.159 | 0.362 | 0.096 | 0.515 | 0.078 | 301 |
| +Eyeglasses, +Smiling | 0.227 | 0.227 | 0.438 | 0.122 | 0.582 | 0.098 | 612 |
| +Black_Hair, -Wavy_Hair | 0.059 | 0.059 | 0.189 | 0.047 | 0.297 | 0.042 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.037 | 0.007 | 0.111 | 0.011 | 27 |
| +Chubby, -Young | 0.173 | 0.173 | 0.409 | 0.092 | 0.521 | 0.066 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.139 | 0.139 | 0.468 | 0.137 | 0.671 | 0.124 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.029 | 0.012 | 0.029 | 0.006 | 34 |
| MACRO | 0.089 | 0.089 | 0.246 | 0.065 | 0.353 | 0.056 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t2_hybrid_tau05_fixed R@1 | naive R@5 | t2_hybrid_tau05_fixed R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.070 | 0.183 | 0.184 |
| +Eyeglasses | 0.018 | 0.119 | 0.077 | 0.395 |
| -Heavy_Makeup | 0.029 | 0.037 | 0.094 | 0.118 |
| +Male | 0.006 | 0.070 | 0.012 | 0.213 |
| -Young | 0.009 | 0.041 | 0.031 | 0.138 |
| +Blond_Hair | 0.024 | 0.067 | 0.082 | 0.213 |
| +Mustache | 0.063 | 0.159 | 0.169 | 0.362 |
| +Eyeglasses, +Smiling | 0.018 | 0.227 | 0.051 | 0.438 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.059 | 0.098 | 0.189 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.037 |
| +Chubby, -Young | 0.003 | 0.173 | 0.012 | 0.409 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.139 | 0.063 | 0.468 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.089 | 0.072 | 0.246 |

## vs sol-A (R@1 / R@5)

| query | sol-A R@1 | t2_hybrid_tau05_fixed R@1 | sol-A R@5 | t2_hybrid_tau05_fixed R@5 |
|---|---|---|---|---|
| +Smiling | 0.061 | 0.070 | 0.177 | 0.184 |
| +Eyeglasses | 0.051 | 0.119 | 0.158 | 0.395 |
| -Heavy_Makeup | 0.030 | 0.037 | 0.103 | 0.118 |
| +Male | 0.008 | 0.070 | 0.018 | 0.213 |
| -Young | 0.012 | 0.041 | 0.034 | 0.138 |
| +Blond_Hair | 0.044 | 0.067 | 0.134 | 0.213 |
| +Mustache | 0.083 | 0.159 | 0.236 | 0.362 |
| +Eyeglasses, +Smiling | 0.064 | 0.227 | 0.160 | 0.438 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.059 | 0.132 | 0.189 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.037 |
| +Chubby, -Young | 0.007 | 0.173 | 0.027 | 0.409 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.139 | 0.190 | 0.468 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.037 | 0.089 | 0.110 | 0.246 |
