# Solution C — Φ-Flow = flow_v2

best sweep cell: N=4, λ=0.0, T=1.0

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.191 | 0.051 | 0.275 | 0.042 | 4786 |
| +Eyeglasses | 0.066 | 0.066 | 0.366 | 0.091 | 0.573 | 0.088 | 2196 |
| -Heavy_Makeup | 0.030 | 0.030 | 0.106 | 0.024 | 0.162 | 0.020 | 4087 |
| +Male | 0.082 | 0.082 | 0.207 | 0.057 | 0.322 | 0.054 | 1595 |
| -Young | 0.045 | 0.045 | 0.139 | 0.036 | 0.216 | 0.033 | 5355 |
| +Blond_Hair | 0.049 | 0.049 | 0.189 | 0.048 | 0.303 | 0.045 | 5469 |
| +Mustache | 0.106 | 0.106 | 0.402 | 0.102 | 0.508 | 0.075 | 301 |
| +Eyeglasses, +Smiling | 0.276 | 0.276 | 0.596 | 0.157 | 0.703 | 0.113 | 612 |
| +Black_Hair, -Wavy_Hair | 0.037 | 0.037 | 0.164 | 0.039 | 0.260 | 0.036 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.037 | 0.007 | 0.148 | 0.022 | 27 |
| +Chubby, -Young | 0.276 | 0.276 | 0.476 | 0.112 | 0.591 | 0.075 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.215 | 0.215 | 0.519 | 0.190 | 0.684 | 0.149 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.000 | 0.029 | 0.003 | 34 |
| MACRO | 0.096 | 0.096 | 0.261 | 0.070 | 0.367 | 0.058 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | flow_v2 R@1 | naive R@5 | flow_v2 R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.070 | 0.183 | 0.191 |
| +Eyeglasses | 0.018 | 0.066 | 0.077 | 0.366 |
| -Heavy_Makeup | 0.029 | 0.030 | 0.094 | 0.106 |
| +Male | 0.006 | 0.082 | 0.012 | 0.207 |
| -Young | 0.009 | 0.045 | 0.031 | 0.139 |
| +Blond_Hair | 0.024 | 0.049 | 0.082 | 0.189 |
| +Mustache | 0.063 | 0.106 | 0.169 | 0.402 |
| +Eyeglasses, +Smiling | 0.018 | 0.276 | 0.051 | 0.596 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.037 | 0.098 | 0.164 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.037 |
| +Chubby, -Young | 0.003 | 0.276 | 0.012 | 0.476 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.215 | 0.063 | 0.519 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.000 |
| MACRO | 0.022 | 0.096 | 0.072 | 0.261 |

## vs t2-hybrid (R@1 / R@5)

| query | t2-hybrid R@1 | flow_v2 R@1 | t2-hybrid R@5 | flow_v2 R@5 |
|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.191 | 0.191 |
| +Eyeglasses | 0.118 | 0.066 | 0.383 | 0.366 |
| -Heavy_Makeup | 0.040 | 0.030 | 0.133 | 0.106 |
| +Male | 0.087 | 0.082 | 0.255 | 0.207 |
| -Young | 0.040 | 0.045 | 0.144 | 0.139 |
| +Blond_Hair | 0.066 | 0.049 | 0.221 | 0.189 |
| +Mustache | 0.116 | 0.106 | 0.306 | 0.402 |
| +Eyeglasses, +Smiling | 0.131 | 0.276 | 0.342 | 0.596 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.037 | 0.171 | 0.164 |
| -Male, -Mustache | 0.074 | 0.000 | 0.148 | 0.037 |
| +Chubby, -Young | 0.149 | 0.276 | 0.363 | 0.476 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.215 | 0.468 | 0.519 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.000 |
| MACRO | 0.090 | 0.096 | 0.240 | 0.261 |
