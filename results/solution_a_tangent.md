# Solution A — tangent-space Log/Exp + dynamic weights (best alpha = 0.8)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.198 | 0.053 | 0.268 | 0.042 | 4786 |
| +Eyeglasses | 0.038 | 0.038 | 0.120 | 0.029 | 0.188 | 0.025 | 2196 |
| -Heavy_Makeup | 0.026 | 0.026 | 0.091 | 0.021 | 0.141 | 0.018 | 4087 |
| +Male | 0.008 | 0.008 | 0.019 | 0.005 | 0.031 | 0.005 | 1595 |
| -Young | 0.009 | 0.009 | 0.028 | 0.007 | 0.046 | 0.006 | 5355 |
| +Blond_Hair | 0.034 | 0.034 | 0.113 | 0.030 | 0.171 | 0.026 | 5469 |
| +Mustache | 0.080 | 0.080 | 0.193 | 0.054 | 0.269 | 0.041 | 301 |
| +Eyeglasses, +Smiling | 0.039 | 0.039 | 0.126 | 0.028 | 0.194 | 0.022 | 612 |
| +Black_Hair, -Wavy_Hair | 0.023 | 0.023 | 0.093 | 0.021 | 0.149 | 0.019 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 27 |
| +Chubby, -Young | 0.005 | 0.005 | 0.012 | 0.003 | 0.024 | 0.003 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.038 | 0.038 | 0.038 | 0.008 | 0.063 | 0.006 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.029 | 0.006 | 0.059 | 0.006 | 34 |
| MACRO | 0.031 | 0.031 | 0.082 | 0.020 | 0.123 | 0.017 | 27697 |

## vs Naive baseline (R@1 / R@5)

| query | naive R@1 | sol-A R@1 | naive R@5 | sol-A R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.070 | 0.183 | 0.198 |
| +Eyeglasses | 0.018 | 0.038 | 0.077 | 0.120 |
| -Heavy_Makeup | 0.029 | 0.026 | 0.094 | 0.091 |
| +Male | 0.006 | 0.008 | 0.012 | 0.019 |
| -Young | 0.009 | 0.009 | 0.031 | 0.028 |
| +Blond_Hair | 0.024 | 0.034 | 0.082 | 0.113 |
| +Mustache | 0.063 | 0.080 | 0.169 | 0.193 |
| +Eyeglasses, +Smiling | 0.018 | 0.039 | 0.051 | 0.126 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.023 | 0.098 | 0.093 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 |
| +Chubby, -Young | 0.003 | 0.005 | 0.012 | 0.012 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.038 | 0.063 | 0.038 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.031 | 0.072 | 0.082 |
