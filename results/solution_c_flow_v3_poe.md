# Solution C — Φ-Flow = flow_v3_poe

PoE re-rank: η=0.0, weighting=none, composition: flow N=8 T=1.0 (train-tuned η, single test run)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.073 | 0.073 | 0.207 | 0.056 | 0.303 | 0.048 | 4786 |
| +Eyeglasses | 0.139 | 0.139 | 0.464 | 0.134 | 0.641 | 0.122 | 2196 |
| -Heavy_Makeup | 0.057 | 0.057 | 0.172 | 0.042 | 0.270 | 0.038 | 4087 |
| +Male | 0.125 | 0.125 | 0.340 | 0.113 | 0.439 | 0.103 | 1595 |
| -Young | 0.046 | 0.046 | 0.180 | 0.045 | 0.280 | 0.043 | 5355 |
| +Blond_Hair | 0.072 | 0.072 | 0.248 | 0.065 | 0.380 | 0.060 | 5469 |
| +Mustache | 0.076 | 0.076 | 0.326 | 0.083 | 0.478 | 0.069 | 301 |
| +Eyeglasses, +Smiling | 0.126 | 0.126 | 0.400 | 0.110 | 0.551 | 0.101 | 612 |
| +Black_Hair, -Wavy_Hair | 0.052 | 0.052 | 0.199 | 0.048 | 0.320 | 0.045 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.185 | 0.037 | 0.333 | 0.033 | 27 |
| +Chubby, -Young | 0.147 | 0.147 | 0.418 | 0.096 | 0.587 | 0.076 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.266 | 0.266 | 0.494 | 0.187 | 0.696 | 0.151 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.029 | 0.006 | 0.059 | 0.006 | 34 |
| MACRO | 0.091 | 0.091 | 0.282 | 0.079 | 0.411 | 0.069 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | flow_v3_poe R@1 | naive R@5 | flow_v3_poe R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.073 | 0.183 | 0.207 |
| +Eyeglasses | 0.018 | 0.139 | 0.077 | 0.464 |
| -Heavy_Makeup | 0.029 | 0.057 | 0.094 | 0.172 |
| +Male | 0.006 | 0.125 | 0.012 | 0.340 |
| -Young | 0.009 | 0.046 | 0.031 | 0.180 |
| +Blond_Hair | 0.024 | 0.072 | 0.082 | 0.248 |
| +Mustache | 0.063 | 0.076 | 0.169 | 0.326 |
| +Eyeglasses, +Smiling | 0.018 | 0.126 | 0.051 | 0.400 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.052 | 0.098 | 0.199 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.185 |
| +Chubby, -Young | 0.003 | 0.147 | 0.012 | 0.418 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.266 | 0.063 | 0.494 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.091 | 0.072 | 0.282 |

## vs t2-hybrid (R@1 / R@5)

| query | t2-hybrid R@1 | flow_v3_poe R@1 | t2-hybrid R@5 | flow_v3_poe R@5 |
|---|---|---|---|---|
| +Smiling | 0.070 | 0.073 | 0.191 | 0.207 |
| +Eyeglasses | 0.118 | 0.139 | 0.383 | 0.464 |
| -Heavy_Makeup | 0.040 | 0.057 | 0.133 | 0.172 |
| +Male | 0.087 | 0.125 | 0.255 | 0.340 |
| -Young | 0.040 | 0.046 | 0.144 | 0.180 |
| +Blond_Hair | 0.066 | 0.072 | 0.221 | 0.248 |
| +Mustache | 0.116 | 0.076 | 0.306 | 0.326 |
| +Eyeglasses, +Smiling | 0.131 | 0.126 | 0.342 | 0.400 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.052 | 0.171 | 0.199 |
| -Male, -Mustache | 0.074 | 0.000 | 0.148 | 0.185 |
| +Chubby, -Young | 0.149 | 0.147 | 0.363 | 0.418 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.266 | 0.468 | 0.494 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.000 | 0.029 |
| MACRO | 0.090 | 0.091 | 0.240 | 0.282 |
