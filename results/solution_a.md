# Solution A — contrastive + dynamic weights (best alpha = 3.0)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.059 | 0.059 | 0.174 | 0.047 | 0.239 | 0.037 | 4786 |
| +Eyeglasses | 0.052 | 0.052 | 0.170 | 0.041 | 0.261 | 0.035 | 2196 |
| -Heavy_Makeup | 0.032 | 0.032 | 0.107 | 0.025 | 0.169 | 0.021 | 4087 |
| +Male | 0.008 | 0.008 | 0.018 | 0.005 | 0.029 | 0.004 | 1595 |
| -Young | 0.011 | 0.011 | 0.032 | 0.008 | 0.053 | 0.007 | 5355 |
| +Blond_Hair | 0.044 | 0.044 | 0.135 | 0.035 | 0.203 | 0.031 | 5469 |
| +Mustache | 0.073 | 0.073 | 0.236 | 0.058 | 0.336 | 0.047 | 301 |
| +Eyeglasses, +Smiling | 0.060 | 0.060 | 0.152 | 0.036 | 0.248 | 0.032 | 612 |
| +Black_Hair, -Wavy_Hair | 0.042 | 0.042 | 0.129 | 0.031 | 0.205 | 0.028 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 27 |
| +Chubby, -Young | 0.007 | 0.007 | 0.024 | 0.005 | 0.041 | 0.004 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.076 | 0.203 | 0.046 | 0.278 | 0.033 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.012 | 0.059 | 0.006 | 34 |
| MACRO | 0.038 | 0.038 | 0.111 | 0.027 | 0.163 | 0.022 | 27697 |

## vs Naive baseline (R@1 / R@5)

| query | naive R@1 | sol-A R@1 | naive R@5 | sol-A R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.059 | 0.183 | 0.174 |
| +Eyeglasses | 0.018 | 0.052 | 0.077 | 0.170 |
| -Heavy_Makeup | 0.029 | 0.032 | 0.094 | 0.107 |
| +Male | 0.006 | 0.008 | 0.012 | 0.018 |
| -Young | 0.009 | 0.011 | 0.031 | 0.032 |
| +Blond_Hair | 0.024 | 0.044 | 0.082 | 0.135 |
| +Mustache | 0.063 | 0.073 | 0.169 | 0.236 |
| +Eyeglasses, +Smiling | 0.018 | 0.060 | 0.051 | 0.152 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.042 | 0.098 | 0.129 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 |
| +Chubby, -Young | 0.003 | 0.007 | 0.012 | 0.024 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.076 | 0.063 | 0.203 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.059 |
| MACRO | 0.022 | 0.038 | 0.072 | 0.111 |
