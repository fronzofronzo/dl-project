# Solution A — T2 learned directions (Φ = t2_hybrid_abl_i2)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.069 | 0.069 | 0.196 | 0.053 | 0.274 | 0.043 | 4786 |
| +Eyeglasses | 0.115 | 0.115 | 0.370 | 0.101 | 0.533 | 0.093 | 2196 |
| -Heavy_Makeup | 0.045 | 0.045 | 0.135 | 0.032 | 0.220 | 0.029 | 4087 |
| +Male | 0.073 | 0.073 | 0.241 | 0.067 | 0.359 | 0.065 | 1595 |
| -Young | 0.039 | 0.039 | 0.145 | 0.035 | 0.224 | 0.033 | 5355 |
| +Blond_Hair | 0.064 | 0.064 | 0.225 | 0.060 | 0.341 | 0.054 | 5469 |
| +Mustache | 0.116 | 0.116 | 0.299 | 0.078 | 0.425 | 0.065 | 301 |
| +Eyeglasses, +Smiling | 0.145 | 0.145 | 0.335 | 0.091 | 0.448 | 0.071 | 612 |
| +Black_Hair, -Wavy_Hair | 0.048 | 0.048 | 0.175 | 0.042 | 0.271 | 0.037 | 2572 |
| -Male, -Mustache | 0.037 | 0.037 | 0.222 | 0.044 | 0.296 | 0.030 | 27 |
| +Chubby, -Young | 0.079 | 0.079 | 0.265 | 0.060 | 0.432 | 0.056 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.215 | 0.215 | 0.443 | 0.142 | 0.658 | 0.123 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.012 | 0.059 | 0.009 | 34 |
| MACRO | 0.083 | 0.083 | 0.239 | 0.063 | 0.349 | 0.054 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t2_hybrid_abl_i2 R@1 | naive R@5 | t2_hybrid_abl_i2 R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.069 | 0.183 | 0.196 |
| +Eyeglasses | 0.018 | 0.115 | 0.077 | 0.370 |
| -Heavy_Makeup | 0.029 | 0.045 | 0.094 | 0.135 |
| +Male | 0.006 | 0.073 | 0.012 | 0.241 |
| -Young | 0.009 | 0.039 | 0.031 | 0.145 |
| +Blond_Hair | 0.024 | 0.064 | 0.082 | 0.225 |
| +Mustache | 0.063 | 0.116 | 0.169 | 0.299 |
| +Eyeglasses, +Smiling | 0.018 | 0.145 | 0.051 | 0.335 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.048 | 0.098 | 0.175 |
| -Male, -Mustache | 0.000 | 0.037 | 0.000 | 0.222 |
| +Chubby, -Young | 0.003 | 0.079 | 0.012 | 0.265 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.215 | 0.063 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.059 |
| MACRO | 0.022 | 0.083 | 0.072 | 0.239 |

## vs sol-A (R@1 / R@5)

| query | sol-A R@1 | t2_hybrid_abl_i2 R@1 | sol-A R@5 | t2_hybrid_abl_i2 R@5 |
|---|---|---|---|---|
| +Smiling | 0.061 | 0.069 | 0.177 | 0.196 |
| +Eyeglasses | 0.051 | 0.115 | 0.158 | 0.370 |
| -Heavy_Makeup | 0.030 | 0.045 | 0.103 | 0.135 |
| +Male | 0.008 | 0.073 | 0.018 | 0.241 |
| -Young | 0.012 | 0.039 | 0.034 | 0.145 |
| +Blond_Hair | 0.044 | 0.064 | 0.134 | 0.225 |
| +Mustache | 0.083 | 0.116 | 0.236 | 0.299 |
| +Eyeglasses, +Smiling | 0.064 | 0.145 | 0.160 | 0.335 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.048 | 0.132 | 0.175 |
| -Male, -Mustache | 0.000 | 0.037 | 0.000 | 0.222 |
| +Chubby, -Young | 0.007 | 0.079 | 0.027 | 0.265 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.215 | 0.190 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.029 | 0.059 | 0.059 |
| MACRO | 0.037 | 0.083 | 0.110 | 0.239 |
