# Solution C — Φ-Flow = poe_only

PoE re-rank: η=0.1, weighting=none, composition: no composition (v_q = v_ref) (train-tuned η, single test run)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.085 | 0.085 | 0.214 | 0.060 | 0.284 | 0.046 | 4786 |
| +Eyeglasses | 0.141 | 0.141 | 0.340 | 0.097 | 0.476 | 0.083 | 2196 |
| -Heavy_Makeup | 0.045 | 0.045 | 0.138 | 0.032 | 0.202 | 0.026 | 4087 |
| +Male | 0.061 | 0.061 | 0.186 | 0.052 | 0.273 | 0.048 | 1595 |
| -Young | 0.021 | 0.021 | 0.074 | 0.018 | 0.128 | 0.017 | 5355 |
| +Blond_Hair | 0.049 | 0.049 | 0.163 | 0.042 | 0.251 | 0.038 | 5469 |
| +Mustache | 0.086 | 0.086 | 0.223 | 0.057 | 0.316 | 0.047 | 301 |
| +Eyeglasses, +Smiling | 0.150 | 0.150 | 0.338 | 0.099 | 0.444 | 0.078 | 612 |
| +Black_Hair, -Wavy_Hair | 0.053 | 0.053 | 0.168 | 0.043 | 0.261 | 0.038 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.074 | 0.015 | 0.148 | 0.019 | 27 |
| +Chubby, -Young | 0.005 | 0.005 | 0.033 | 0.007 | 0.062 | 0.007 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.177 | 0.177 | 0.405 | 0.124 | 0.608 | 0.119 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.018 | 0.059 | 0.009 | 34 |
| MACRO | 0.069 | 0.069 | 0.186 | 0.051 | 0.270 | 0.044 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | poe_only R@1 | naive R@5 | poe_only R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.085 | 0.183 | 0.214 |
| +Eyeglasses | 0.018 | 0.141 | 0.077 | 0.340 |
| -Heavy_Makeup | 0.029 | 0.045 | 0.094 | 0.138 |
| +Male | 0.006 | 0.061 | 0.012 | 0.186 |
| -Young | 0.009 | 0.021 | 0.031 | 0.074 |
| +Blond_Hair | 0.024 | 0.049 | 0.082 | 0.163 |
| +Mustache | 0.063 | 0.086 | 0.169 | 0.223 |
| +Eyeglasses, +Smiling | 0.018 | 0.150 | 0.051 | 0.338 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.053 | 0.098 | 0.168 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.074 |
| +Chubby, -Young | 0.003 | 0.005 | 0.012 | 0.033 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.177 | 0.063 | 0.405 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.059 |
| MACRO | 0.022 | 0.069 | 0.072 | 0.186 |

## vs t2-hybrid (R@1 / R@5)

| query | t2-hybrid R@1 | poe_only R@1 | t2-hybrid R@5 | poe_only R@5 |
|---|---|---|---|---|
| +Smiling | 0.070 | 0.085 | 0.191 | 0.214 |
| +Eyeglasses | 0.118 | 0.141 | 0.383 | 0.340 |
| -Heavy_Makeup | 0.040 | 0.045 | 0.133 | 0.138 |
| +Male | 0.087 | 0.061 | 0.255 | 0.186 |
| -Young | 0.040 | 0.021 | 0.144 | 0.074 |
| +Blond_Hair | 0.066 | 0.049 | 0.221 | 0.163 |
| +Mustache | 0.116 | 0.086 | 0.306 | 0.223 |
| +Eyeglasses, +Smiling | 0.131 | 0.150 | 0.342 | 0.338 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.053 | 0.171 | 0.168 |
| -Male, -Mustache | 0.074 | 0.000 | 0.148 | 0.074 |
| +Chubby, -Young | 0.149 | 0.005 | 0.363 | 0.033 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.177 | 0.468 | 0.405 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.029 | 0.000 | 0.059 |
| MACRO | 0.090 | 0.069 | 0.240 | 0.186 |
