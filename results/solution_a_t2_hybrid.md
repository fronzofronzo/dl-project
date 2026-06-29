# Solution A — T2 learned directions (Φ = t2_hybrid)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.191 | 0.051 | 0.269 | 0.042 | 4786 |
| +Eyeglasses | 0.118 | 0.118 | 0.383 | 0.102 | 0.567 | 0.097 | 2196 |
| -Heavy_Makeup | 0.040 | 0.040 | 0.133 | 0.031 | 0.212 | 0.028 | 4087 |
| +Male | 0.087 | 0.087 | 0.255 | 0.074 | 0.354 | 0.069 | 1595 |
| -Young | 0.040 | 0.040 | 0.144 | 0.035 | 0.228 | 0.033 | 5355 |
| +Blond_Hair | 0.066 | 0.066 | 0.221 | 0.058 | 0.335 | 0.053 | 5469 |
| +Mustache | 0.116 | 0.116 | 0.306 | 0.084 | 0.402 | 0.065 | 301 |
| +Eyeglasses, +Smiling | 0.131 | 0.131 | 0.342 | 0.082 | 0.461 | 0.066 | 612 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.050 | 0.171 | 0.042 | 0.269 | 0.038 | 2572 |
| -Male, -Mustache | 0.074 | 0.074 | 0.148 | 0.037 | 0.259 | 0.033 | 27 |
| +Chubby, -Young | 0.149 | 0.149 | 0.363 | 0.083 | 0.476 | 0.061 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.228 | 0.468 | 0.152 | 0.671 | 0.124 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.000 | 0.029 | 0.003 | 34 |
| MACRO | 0.090 | 0.090 | 0.240 | 0.064 | 0.349 | 0.055 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t2_hybrid R@1 | naive R@5 | t2_hybrid R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.070 | 0.183 | 0.191 |
| +Eyeglasses | 0.018 | 0.118 | 0.077 | 0.383 |
| -Heavy_Makeup | 0.029 | 0.040 | 0.094 | 0.133 |
| +Male | 0.006 | 0.087 | 0.012 | 0.255 |
| -Young | 0.009 | 0.040 | 0.031 | 0.144 |
| +Blond_Hair | 0.024 | 0.066 | 0.082 | 0.221 |
| +Mustache | 0.063 | 0.116 | 0.169 | 0.306 |
| +Eyeglasses, +Smiling | 0.018 | 0.131 | 0.051 | 0.342 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.050 | 0.098 | 0.171 |
| -Male, -Mustache | 0.000 | 0.074 | 0.000 | 0.148 |
| +Chubby, -Young | 0.003 | 0.149 | 0.012 | 0.363 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.228 | 0.063 | 0.468 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.000 |
| MACRO | 0.022 | 0.090 | 0.072 | 0.240 |

## vs sol-A (R@1 / R@5)

| query | sol-A R@1 | t2_hybrid R@1 | sol-A R@5 | t2_hybrid R@5 |
|---|---|---|---|---|
| +Smiling | 0.061 | 0.070 | 0.177 | 0.191 |
| +Eyeglasses | 0.051 | 0.118 | 0.158 | 0.383 |
| -Heavy_Makeup | 0.030 | 0.040 | 0.103 | 0.133 |
| +Male | 0.008 | 0.087 | 0.018 | 0.255 |
| -Young | 0.012 | 0.040 | 0.034 | 0.144 |
| +Blond_Hair | 0.044 | 0.066 | 0.134 | 0.221 |
| +Mustache | 0.083 | 0.116 | 0.236 | 0.306 |
| +Eyeglasses, +Smiling | 0.064 | 0.131 | 0.160 | 0.342 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.050 | 0.132 | 0.171 |
| -Male, -Mustache | 0.000 | 0.074 | 0.000 | 0.148 |
| +Chubby, -Young | 0.007 | 0.149 | 0.027 | 0.363 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.228 | 0.190 | 0.468 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.059 | 0.000 |
| MACRO | 0.037 | 0.090 | 0.110 | 0.240 |
