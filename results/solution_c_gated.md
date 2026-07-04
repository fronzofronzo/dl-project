# Solution C — Φ-Flow = gated

gate: |C|>=2 -> flow(N=8, T=1.0), else t2-hybrid

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.191 | 0.051 | 0.269 | 0.042 | 4786 |
| +Eyeglasses | 0.118 | 0.118 | 0.383 | 0.102 | 0.567 | 0.097 | 2196 |
| -Heavy_Makeup | 0.040 | 0.040 | 0.133 | 0.031 | 0.212 | 0.028 | 4087 |
| +Male | 0.087 | 0.087 | 0.255 | 0.074 | 0.354 | 0.069 | 1595 |
| -Young | 0.040 | 0.040 | 0.144 | 0.035 | 0.228 | 0.033 | 5355 |
| +Blond_Hair | 0.066 | 0.066 | 0.221 | 0.058 | 0.335 | 0.053 | 5469 |
| +Mustache | 0.116 | 0.116 | 0.306 | 0.084 | 0.402 | 0.065 | 301 |
| +Eyeglasses, +Smiling | 0.126 | 0.126 | 0.400 | 0.110 | 0.551 | 0.101 | 612 |
| +Black_Hair, -Wavy_Hair | 0.052 | 0.052 | 0.199 | 0.048 | 0.320 | 0.045 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.185 | 0.037 | 0.333 | 0.033 | 27 |
| +Chubby, -Young | 0.147 | 0.147 | 0.418 | 0.096 | 0.587 | 0.076 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.266 | 0.266 | 0.494 | 0.187 | 0.696 | 0.151 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.029 | 0.006 | 0.059 | 0.006 | 34 |
| MACRO | 0.087 | 0.087 | 0.258 | 0.071 | 0.378 | 0.061 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | gated R@1 | naive R@5 | gated R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.070 | 0.183 | 0.191 |
| +Eyeglasses | 0.018 | 0.118 | 0.077 | 0.383 |
| -Heavy_Makeup | 0.029 | 0.040 | 0.094 | 0.133 |
| +Male | 0.006 | 0.087 | 0.012 | 0.255 |
| -Young | 0.009 | 0.040 | 0.031 | 0.144 |
| +Blond_Hair | 0.024 | 0.066 | 0.082 | 0.221 |
| +Mustache | 0.063 | 0.116 | 0.169 | 0.306 |
| +Eyeglasses, +Smiling | 0.018 | 0.126 | 0.051 | 0.400 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.052 | 0.098 | 0.199 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.185 |
| +Chubby, -Young | 0.003 | 0.147 | 0.012 | 0.418 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.266 | 0.063 | 0.494 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.087 | 0.072 | 0.258 |

## vs t2-hybrid (R@1 / R@5)

| query | t2-hybrid R@1 | gated R@1 | t2-hybrid R@5 | gated R@5 |
|---|---|---|---|---|
| +Smiling | 0.070 | 0.070 | 0.191 | 0.191 |
| +Eyeglasses | 0.118 | 0.118 | 0.383 | 0.383 |
| -Heavy_Makeup | 0.040 | 0.040 | 0.133 | 0.133 |
| +Male | 0.087 | 0.087 | 0.255 | 0.255 |
| -Young | 0.040 | 0.040 | 0.144 | 0.144 |
| +Blond_Hair | 0.066 | 0.066 | 0.221 | 0.221 |
| +Mustache | 0.116 | 0.116 | 0.306 | 0.306 |
| +Eyeglasses, +Smiling | 0.131 | 0.126 | 0.342 | 0.400 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.052 | 0.171 | 0.199 |
| -Male, -Mustache | 0.074 | 0.000 | 0.148 | 0.185 |
| +Chubby, -Young | 0.149 | 0.147 | 0.363 | 0.418 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.266 | 0.468 | 0.494 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.029 |
| MACRO | 0.090 | 0.087 | 0.240 | 0.258 |
