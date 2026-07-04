# Solution C — Φ-Flow = flow_main

best sweep cell: N=8, λ=0.0, T=1.0

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.087 | 0.087 | 0.227 | 0.061 | 0.316 | 0.050 | 4786 |
| +Eyeglasses | 0.160 | 0.160 | 0.446 | 0.132 | 0.597 | 0.116 | 2196 |
| -Heavy_Makeup | 0.045 | 0.045 | 0.152 | 0.036 | 0.237 | 0.031 | 4087 |
| +Male | 0.094 | 0.094 | 0.278 | 0.081 | 0.387 | 0.074 | 1595 |
| -Young | 0.041 | 0.041 | 0.141 | 0.035 | 0.228 | 0.033 | 5355 |
| +Blond_Hair | 0.070 | 0.070 | 0.235 | 0.063 | 0.357 | 0.057 | 5469 |
| +Mustache | 0.110 | 0.110 | 0.342 | 0.092 | 0.452 | 0.072 | 301 |
| +Eyeglasses, +Smiling | 0.124 | 0.124 | 0.307 | 0.090 | 0.456 | 0.076 | 612 |
| +Black_Hair, -Wavy_Hair | 0.068 | 0.068 | 0.198 | 0.049 | 0.301 | 0.043 | 2572 |
| -Male, -Mustache | 0.037 | 0.037 | 0.074 | 0.015 | 0.148 | 0.022 | 27 |
| +Chubby, -Young | 0.060 | 0.060 | 0.231 | 0.052 | 0.360 | 0.045 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.253 | 0.253 | 0.532 | 0.165 | 0.747 | 0.130 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.000 | 0.059 | 0.006 | 34 |
| MACRO | 0.088 | 0.088 | 0.243 | 0.067 | 0.357 | 0.058 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | flow_main R@1 | naive R@5 | flow_main R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.087 | 0.183 | 0.227 |
| +Eyeglasses | 0.018 | 0.160 | 0.077 | 0.446 |
| -Heavy_Makeup | 0.029 | 0.045 | 0.094 | 0.152 |
| +Male | 0.006 | 0.094 | 0.012 | 0.278 |
| -Young | 0.009 | 0.041 | 0.031 | 0.141 |
| +Blond_Hair | 0.024 | 0.070 | 0.082 | 0.235 |
| +Mustache | 0.063 | 0.110 | 0.169 | 0.342 |
| +Eyeglasses, +Smiling | 0.018 | 0.124 | 0.051 | 0.307 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.068 | 0.098 | 0.198 |
| -Male, -Mustache | 0.000 | 0.037 | 0.000 | 0.074 |
| +Chubby, -Young | 0.003 | 0.060 | 0.012 | 0.231 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.253 | 0.063 | 0.532 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.000 |
| MACRO | 0.022 | 0.088 | 0.072 | 0.243 |

## vs t2-hybrid (R@1 / R@5)

| query | t2-hybrid R@1 | flow_main R@1 | t2-hybrid R@5 | flow_main R@5 |
|---|---|---|---|---|
| +Smiling | 0.070 | 0.087 | 0.191 | 0.227 |
| +Eyeglasses | 0.118 | 0.160 | 0.383 | 0.446 |
| -Heavy_Makeup | 0.040 | 0.045 | 0.133 | 0.152 |
| +Male | 0.087 | 0.094 | 0.255 | 0.278 |
| -Young | 0.040 | 0.041 | 0.144 | 0.141 |
| +Blond_Hair | 0.066 | 0.070 | 0.221 | 0.235 |
| +Mustache | 0.116 | 0.110 | 0.306 | 0.342 |
| +Eyeglasses, +Smiling | 0.131 | 0.124 | 0.342 | 0.307 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.068 | 0.171 | 0.198 |
| -Male, -Mustache | 0.074 | 0.037 | 0.148 | 0.074 |
| +Chubby, -Young | 0.149 | 0.060 | 0.363 | 0.231 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.253 | 0.468 | 0.532 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.000 |
| MACRO | 0.090 | 0.088 | 0.240 | 0.243 |
