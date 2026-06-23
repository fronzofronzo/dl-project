# Solution A — contrastive + negative re-rank (best alpha=4.0, lambda=0.0, orth=conflict)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.061 | 0.061 | 0.177 | 0.047 | 0.242 | 0.038 | 4786 |
| +Eyeglasses | 0.051 | 0.051 | 0.158 | 0.038 | 0.238 | 0.032 | 2196 |
| -Heavy_Makeup | 0.030 | 0.030 | 0.103 | 0.023 | 0.166 | 0.020 | 4087 |
| +Male | 0.008 | 0.008 | 0.018 | 0.005 | 0.026 | 0.004 | 1595 |
| -Young | 0.012 | 0.012 | 0.034 | 0.008 | 0.056 | 0.007 | 5355 |
| +Blond_Hair | 0.044 | 0.044 | 0.134 | 0.034 | 0.205 | 0.031 | 5469 |
| +Mustache | 0.083 | 0.083 | 0.236 | 0.060 | 0.336 | 0.047 | 301 |
| +Eyeglasses, +Smiling | 0.064 | 0.064 | 0.160 | 0.037 | 0.242 | 0.031 | 612 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.040 | 0.132 | 0.032 | 0.199 | 0.027 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 27 |
| +Chubby, -Young | 0.007 | 0.007 | 0.027 | 0.006 | 0.045 | 0.005 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.076 | 0.190 | 0.046 | 0.278 | 0.034 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.059 | 0.012 | 0.059 | 0.006 | 34 |
| MACRO | 0.037 | 0.037 | 0.110 | 0.027 | 0.161 | 0.022 | 27697 |

## vs Naive baseline (R@1 / R@5)

| query | naive R@1 | sol-A R@1 | naive R@5 | sol-A R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.061 | 0.183 | 0.177 |
| +Eyeglasses | 0.018 | 0.051 | 0.077 | 0.158 |
| -Heavy_Makeup | 0.029 | 0.030 | 0.094 | 0.103 |
| +Male | 0.006 | 0.008 | 0.012 | 0.018 |
| -Young | 0.009 | 0.012 | 0.031 | 0.034 |
| +Blond_Hair | 0.024 | 0.044 | 0.082 | 0.134 |
| +Mustache | 0.063 | 0.083 | 0.169 | 0.236 |
| +Eyeglasses, +Smiling | 0.018 | 0.064 | 0.051 | 0.160 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.040 | 0.098 | 0.132 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 |
| +Chubby, -Young | 0.003 | 0.007 | 0.012 | 0.027 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.076 | 0.063 | 0.190 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.059 |
| MACRO | 0.022 | 0.037 | 0.072 | 0.110 |
