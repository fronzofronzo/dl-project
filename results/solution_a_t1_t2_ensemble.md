# Solution A — T2 learned directions (Φ = t1_t2_ensemble)

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.076 | 0.076 | 0.205 | 0.056 | 0.284 | 0.045 | 4786 |
| +Eyeglasses | 0.107 | 0.107 | 0.362 | 0.093 | 0.520 | 0.086 | 2196 |
| -Heavy_Makeup | 0.045 | 0.045 | 0.133 | 0.032 | 0.216 | 0.028 | 4087 |
| +Male | 0.068 | 0.068 | 0.224 | 0.064 | 0.334 | 0.060 | 1595 |
| -Young | 0.033 | 0.033 | 0.140 | 0.033 | 0.228 | 0.031 | 5355 |
| +Blond_Hair | 0.065 | 0.065 | 0.222 | 0.058 | 0.329 | 0.052 | 5469 |
| +Mustache | 0.120 | 0.120 | 0.322 | 0.086 | 0.435 | 0.069 | 301 |
| +Eyeglasses, +Smiling | 0.106 | 0.106 | 0.306 | 0.072 | 0.446 | 0.059 | 612 |
| +Black_Hair, -Wavy_Hair | 0.059 | 0.059 | 0.193 | 0.047 | 0.284 | 0.041 | 2572 |
| -Male, -Mustache | 0.037 | 0.037 | 0.074 | 0.015 | 0.222 | 0.022 | 27 |
| +Chubby, -Young | 0.295 | 0.295 | 0.469 | 0.100 | 0.574 | 0.067 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.165 | 0.165 | 0.443 | 0.127 | 0.646 | 0.110 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.029 | 0.006 | 0.059 | 0.006 | 34 |
| MACRO | 0.090 | 0.090 | 0.240 | 0.061 | 0.352 | 0.052 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | t1_t2_ensemble R@1 | naive R@5 | t1_t2_ensemble R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.076 | 0.183 | 0.205 |
| +Eyeglasses | 0.018 | 0.107 | 0.077 | 0.362 |
| -Heavy_Makeup | 0.029 | 0.045 | 0.094 | 0.133 |
| +Male | 0.006 | 0.068 | 0.012 | 0.224 |
| -Young | 0.009 | 0.033 | 0.031 | 0.140 |
| +Blond_Hair | 0.024 | 0.065 | 0.082 | 0.222 |
| +Mustache | 0.063 | 0.120 | 0.169 | 0.322 |
| +Eyeglasses, +Smiling | 0.018 | 0.106 | 0.051 | 0.306 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.059 | 0.098 | 0.193 |
| -Male, -Mustache | 0.000 | 0.037 | 0.000 | 0.074 |
| +Chubby, -Young | 0.003 | 0.295 | 0.012 | 0.469 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.165 | 0.063 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.022 | 0.090 | 0.072 | 0.240 |

## vs sol-A (R@1 / R@5)

| query | sol-A R@1 | t1_t2_ensemble R@1 | sol-A R@5 | t1_t2_ensemble R@5 |
|---|---|---|---|---|
| +Smiling | 0.061 | 0.076 | 0.177 | 0.205 |
| +Eyeglasses | 0.051 | 0.107 | 0.158 | 0.362 |
| -Heavy_Makeup | 0.030 | 0.045 | 0.103 | 0.133 |
| +Male | 0.008 | 0.068 | 0.018 | 0.224 |
| -Young | 0.012 | 0.033 | 0.034 | 0.140 |
| +Blond_Hair | 0.044 | 0.065 | 0.134 | 0.222 |
| +Mustache | 0.083 | 0.120 | 0.236 | 0.322 |
| +Eyeglasses, +Smiling | 0.064 | 0.106 | 0.160 | 0.306 |
| +Black_Hair, -Wavy_Hair | 0.040 | 0.059 | 0.132 | 0.193 |
| -Male, -Mustache | 0.000 | 0.037 | 0.000 | 0.074 |
| +Chubby, -Young | 0.007 | 0.295 | 0.027 | 0.469 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.076 | 0.165 | 0.190 | 0.443 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.000 | 0.059 | 0.029 |
| MACRO | 0.037 | 0.090 | 0.110 | 0.240 |
