# Solution A — T2 learned directions (Φ = t2_hybrid_abl_i3)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.068 | 0.068 | 0.179 | 0.048 | 0.251 | 0.039 | 4786 |
| +Eyeglasses | 0.104 | 0.104 | 0.344 | 0.089 | 0.505 | 0.083 | 2196 |
| -Heavy_Makeup | 0.038 | 0.038 | 0.131 | 0.031 | 0.204 | 0.027 | 4087 |
| +Male | 0.066 | 0.066 | 0.193 | 0.051 | 0.288 | 0.048 | 1595 |
| -Young | 0.038 | 0.038 | 0.119 | 0.029 | 0.194 | 0.027 | 5355 |
| +Blond_Hair | 0.071 | 0.071 | 0.224 | 0.059 | 0.338 | 0.054 | 5469 |
| +Mustache | 0.100 | 0.100 | 0.296 | 0.078 | 0.425 | 0.067 | 301 |
| +Eyeglasses, +Smiling | 0.132 | 0.132 | 0.315 | 0.086 | 0.436 | 0.073 | 612 |
| +Black_Hair, -Wavy_Hair | 0.047 | 0.047 | 0.163 | 0.040 | 0.253 | 0.036 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.074 | 0.015 | 0.148 | 0.022 | 27 |
| +Chubby, -Young | 0.096 | 0.096 | 0.348 | 0.074 | 0.476 | 0.058 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.253 | 0.253 | 0.532 | 0.165 | 0.797 | 0.141 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.029 | 0.006 | 0.059 | 0.009 | 34 |
| MACRO | 0.080 | 0.080 | 0.227 | 0.059 | 0.337 | 0.052 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t2_hybrid_abl_i3 R@1 | naive R@5 | t2_hybrid_abl_i3 R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.068 | 0.183 | 0.179 |
| +Eyeglasses | 0.018 | 0.104 | 0.077 | 0.344 |
| -Heavy_Makeup | 0.029 | 0.038 | 0.094 | 0.131 |
| +Male | 0.006 | 0.066 | 0.012 | 0.193 |
| -Young | 0.009 | 0.038 | 0.031 | 0.119 |
| +Blond_Hair | 0.024 | 0.071 | 0.082 | 0.224 |
| +Mustache | 0.063 | 0.100 | 0.169 | 0.296 |
| +Eyeglasses, +Smiling | 0.018 | 0.132 | 0.051 | 0.315 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.047 | 0.098 | 0.163 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.074 |
| +Chubby, -Young | 0.003 | 0.096 | 0.012 | 0.348 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.253 | 0.063 | 0.532 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.080 | 0.072 | 0.227 |

## vs sol-A (R@1 / R@5)

| query | sol-A R@1 | t2_hybrid_abl_i3 R@1 | sol-A R@5 | t2_hybrid_abl_i3 R@5 |
|---|---|---|---|---|
| +Smiling | 0.061 | 0.068 | 0.177 | 0.179 |
| +Eyeglasses | 0.051 | 0.104 | 0.158 | 0.344 |
| -Heavy_Makeup | 0.030 | 0.038 | 0.103 | 0.131 |
| +Male | 0.008 | 0.066 | 0.018 | 0.193 |
| -Young | 0.012 | 0.038 | 0.034 | 0.119 |
| +Blond_Hair | 0.044 | 0.071 | 0.134 | 0.224 |
| +Mustache | 0.083 | 0.100 | 0.236 | 0.296 |
| +Eyeglasses, +Smiling | 0.064 | 0.132 | 0.160 | 0.315 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.047 | 0.132 | 0.163 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.074 |
| +Chubby, -Young | 0.007 | 0.096 | 0.027 | 0.348 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.253 | 0.190 | 0.532 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.029 | 0.059 | 0.029 |
| MACRO | 0.037 | 0.080 | 0.110 | 0.227 |
