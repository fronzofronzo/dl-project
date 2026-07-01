# Solution A — T2 learned directions (Φ = t2_hybrid_abl_i1)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.073 | 0.073 | 0.197 | 0.052 | 0.268 | 0.042 | 4786 |
| +Eyeglasses | 0.115 | 0.115 | 0.306 | 0.082 | 0.460 | 0.075 | 2196 |
| -Heavy_Makeup | 0.029 | 0.029 | 0.104 | 0.023 | 0.161 | 0.021 | 4087 |
| +Male | 0.050 | 0.050 | 0.161 | 0.041 | 0.244 | 0.037 | 1595 |
| -Young | 0.019 | 0.019 | 0.070 | 0.016 | 0.126 | 0.016 | 5355 |
| +Blond_Hair | 0.056 | 0.056 | 0.186 | 0.050 | 0.282 | 0.044 | 5469 |
| +Mustache | 0.136 | 0.136 | 0.326 | 0.082 | 0.402 | 0.062 | 301 |
| +Eyeglasses, +Smiling | 0.109 | 0.109 | 0.337 | 0.083 | 0.467 | 0.068 | 612 |
| +Black_Hair, -Wavy_Hair | 0.038 | 0.038 | 0.135 | 0.033 | 0.215 | 0.029 | 2572 |
| -Male, -Mustache | 0.000 | 0.000 | 0.037 | 0.015 | 0.074 | 0.019 | 27 |
| +Chubby, -Young | 0.019 | 0.019 | 0.127 | 0.027 | 0.255 | 0.029 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.203 | 0.203 | 0.443 | 0.144 | 0.557 | 0.124 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.029 | 0.006 | 0.029 | 0.006 | 34 |
| MACRO | 0.065 | 0.065 | 0.189 | 0.050 | 0.272 | 0.044 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t2_hybrid_abl_i1 R@1 | naive R@5 | t2_hybrid_abl_i1 R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.073 | 0.183 | 0.197 |
| +Eyeglasses | 0.018 | 0.115 | 0.077 | 0.306 |
| -Heavy_Makeup | 0.029 | 0.029 | 0.094 | 0.104 |
| +Male | 0.006 | 0.050 | 0.012 | 0.161 |
| -Young | 0.009 | 0.019 | 0.031 | 0.070 |
| +Blond_Hair | 0.024 | 0.056 | 0.082 | 0.186 |
| +Mustache | 0.063 | 0.136 | 0.169 | 0.326 |
| +Eyeglasses, +Smiling | 0.018 | 0.109 | 0.051 | 0.337 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.038 | 0.098 | 0.135 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.037 |
| +Chubby, -Young | 0.003 | 0.019 | 0.012 | 0.127 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.203 | 0.063 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.065 | 0.072 | 0.189 |

## vs sol-A (R@1 / R@5)

| query | sol-A R@1 | t2_hybrid_abl_i1 R@1 | sol-A R@5 | t2_hybrid_abl_i1 R@5 |
|---|---|---|---|---|
| +Smiling | 0.061 | 0.073 | 0.177 | 0.197 |
| +Eyeglasses | 0.051 | 0.115 | 0.158 | 0.306 |
| -Heavy_Makeup | 0.030 | 0.029 | 0.103 | 0.104 |
| +Male | 0.008 | 0.050 | 0.018 | 0.161 |
| -Young | 0.012 | 0.019 | 0.034 | 0.070 |
| +Blond_Hair | 0.044 | 0.056 | 0.134 | 0.186 |
| +Mustache | 0.083 | 0.136 | 0.236 | 0.326 |
| +Eyeglasses, +Smiling | 0.064 | 0.109 | 0.160 | 0.337 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.038 | 0.132 | 0.135 |
| -Male, -Mustache | 0.000 | 0.000 | 0.000 | 0.037 |
| +Chubby, -Young | 0.007 | 0.019 | 0.027 | 0.127 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.203 | 0.190 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.037 | 0.065 | 0.110 | 0.189 |
