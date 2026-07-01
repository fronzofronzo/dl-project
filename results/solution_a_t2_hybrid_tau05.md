# Solution A — T2 learned directions (Φ = t2_hybrid_tau05)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.069 | 0.069 | 0.187 | 0.051 | 0.266 | 0.042 | 4786 |
| +Eyeglasses | 0.105 | 0.105 | 0.360 | 0.094 | 0.524 | 0.088 | 2196 |
| -Heavy_Makeup | 0.037 | 0.037 | 0.124 | 0.029 | 0.196 | 0.025 | 4087 |
| +Male | 0.066 | 0.066 | 0.204 | 0.055 | 0.318 | 0.054 | 1595 |
| -Young | 0.043 | 0.043 | 0.142 | 0.036 | 0.222 | 0.033 | 5355 |
| +Blond_Hair | 0.063 | 0.063 | 0.213 | 0.055 | 0.329 | 0.051 | 5469 |
| +Mustache | 0.140 | 0.140 | 0.372 | 0.094 | 0.515 | 0.074 | 301 |
| +Eyeglasses, +Smiling | 0.194 | 0.194 | 0.395 | 0.110 | 0.528 | 0.088 | 612 |
| +Black_Hair, -Wavy_Hair | 0.054 | 0.054 | 0.177 | 0.043 | 0.276 | 0.038 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 | 0.148 | 0.015 | 27 |
| +Chubby, -Young | 0.139 | 0.139 | 0.416 | 0.090 | 0.551 | 0.067 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.241 | 0.241 | 0.443 | 0.152 | 0.633 | 0.129 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.029 | 0.012 | 0.118 | 0.015 | 34 |
| MACRO | 0.089 | 0.089 | 0.236 | 0.063 | 0.356 | 0.055 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t2_hybrid_tau05 R@1 | naive R@5 | t2_hybrid_tau05 R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.069 | 0.183 | 0.187 |
| +Eyeglasses | 0.018 | 0.105 | 0.077 | 0.360 |
| -Heavy_Makeup | 0.029 | 0.037 | 0.094 | 0.124 |
| +Male | 0.006 | 0.066 | 0.012 | 0.204 |
| -Young | 0.009 | 0.043 | 0.031 | 0.142 |
| +Blond_Hair | 0.024 | 0.063 | 0.082 | 0.213 |
| +Mustache | 0.063 | 0.140 | 0.169 | 0.372 |
| +Eyeglasses, +Smiling | 0.018 | 0.194 | 0.051 | 0.395 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.054 | 0.098 | 0.177 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 |
| +Chubby, -Young | 0.003 | 0.139 | 0.012 | 0.416 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.241 | 0.063 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.089 | 0.072 | 0.236 |

## vs sol-A (R@1 / R@5)

| query | sol-A R@1 | t2_hybrid_tau05 R@1 | sol-A R@5 | t2_hybrid_tau05 R@5 |
|---|---|---|---|---|
| +Smiling | 0.061 | 0.069 | 0.177 | 0.187 |
| +Eyeglasses | 0.051 | 0.105 | 0.158 | 0.360 |
| -Heavy_Makeup | 0.030 | 0.037 | 0.103 | 0.124 |
| +Male | 0.008 | 0.066 | 0.018 | 0.204 |
| -Young | 0.012 | 0.043 | 0.034 | 0.142 |
| +Blond_Hair | 0.044 | 0.063 | 0.134 | 0.213 |
| +Mustache | 0.083 | 0.140 | 0.236 | 0.372 |
| +Eyeglasses, +Smiling | 0.064 | 0.194 | 0.160 | 0.395 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.054 | 0.132 | 0.177 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.000 |
| +Chubby, -Young | 0.007 | 0.139 | 0.027 | 0.416 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.241 | 0.190 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.037 | 0.089 | 0.110 | 0.236 |
