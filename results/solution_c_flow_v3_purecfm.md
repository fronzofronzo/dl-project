# Solution C — Φ-Flow = flow_v3_purecfm

best sweep cell: N=4, λ=0.0, T=1.0

| query | recall@1 | precision@1 | recall@5 | precision@5 | recall@10 | precision@10 | n_sources |
|---|---|---|---|---|---|---|---|
| +Smiling | 0.058 | 0.058 | 0.184 | 0.048 | 0.258 | 0.039 | 4786 |
| +Eyeglasses | 0.031 | 0.031 | 0.118 | 0.028 | 0.179 | 0.024 | 2196 |
| -Heavy_Makeup | 0.033 | 0.033 | 0.110 | 0.026 | 0.168 | 0.021 | 4087 |
| +Male | 0.047 | 0.047 | 0.150 | 0.040 | 0.243 | 0.037 | 1595 |
| -Young | 0.013 | 0.013 | 0.046 | 0.011 | 0.085 | 0.011 | 5355 |
| +Blond_Hair | 0.034 | 0.034 | 0.118 | 0.030 | 0.195 | 0.029 | 5469 |
| +Mustache | 0.073 | 0.073 | 0.153 | 0.043 | 0.233 | 0.034 | 301 |
| +Eyeglasses, +Smiling | 0.029 | 0.029 | 0.085 | 0.019 | 0.142 | 0.017 | 612 |
| +Black_Hair, -Wavy_Hair | 0.019 | 0.019 | 0.082 | 0.019 | 0.140 | 0.018 | 2572 |
| -Male, -Mustache | 0.074 | 0.074 | 0.074 | 0.030 | 0.148 | 0.030 | 27 |
| +Chubby, -Young | 0.005 | 0.005 | 0.026 | 0.005 | 0.053 | 0.006 | 584 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.013 | 0.013 | 0.013 | 0.003 | 0.063 | 0.006 | 79 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.018 | 0.088 | 0.012 | 34 |
| MACRO | 0.035 | 0.035 | 0.094 | 0.025 | 0.153 | 0.022 | 27697 |

## vs naive (R@1 / R@5)

| query | naive R@1 | flow_v3_purecfm R@1 | naive R@5 | flow_v3_purecfm R@5 |
|---|---|---|---|---|
| +Smiling | 0.065 | 0.058 | 0.183 | 0.184 |
| +Eyeglasses | 0.018 | 0.031 | 0.077 | 0.118 |
| -Heavy_Makeup | 0.029 | 0.033 | 0.094 | 0.110 |
| +Male | 0.006 | 0.047 | 0.012 | 0.150 |
| -Young | 0.009 | 0.013 | 0.031 | 0.046 |
| +Blond_Hair | 0.024 | 0.034 | 0.082 | 0.118 |
| +Mustache | 0.063 | 0.073 | 0.169 | 0.153 |
| +Eyeglasses, +Smiling | 0.018 | 0.029 | 0.051 | 0.085 |
| +Black_Hair, -Wavy_Hair | 0.026 | 0.019 | 0.098 | 0.082 |
| -Male, -Mustache | 0.000 | 0.074 | 0.000 | 0.074 |
| +Chubby, -Young | 0.003 | 0.005 | 0.012 | 0.026 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.000 | 0.013 | 0.063 | 0.013 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.029 | 0.029 | 0.059 | 0.059 |
| MACRO | 0.022 | 0.035 | 0.072 | 0.094 |

## vs t2-hybrid (R@1 / R@5)

| query | t2-hybrid R@1 | flow_v3_purecfm R@1 | t2-hybrid R@5 | flow_v3_purecfm R@5 |
|---|---|---|---|---|
| +Smiling | 0.070 | 0.058 | 0.191 | 0.184 |
| +Eyeglasses | 0.118 | 0.031 | 0.383 | 0.118 |
| -Heavy_Makeup | 0.040 | 0.033 | 0.133 | 0.110 |
| +Male | 0.087 | 0.047 | 0.255 | 0.150 |
| -Young | 0.040 | 0.013 | 0.144 | 0.046 |
| +Blond_Hair | 0.066 | 0.034 | 0.221 | 0.118 |
| +Mustache | 0.116 | 0.073 | 0.306 | 0.153 |
| +Eyeglasses, +Smiling | 0.131 | 0.029 | 0.342 | 0.085 |
| +Black_Hair, -Wavy_Hair | 0.050 | 0.019 | 0.171 | 0.082 |
| -Male, -Mustache | 0.074 | 0.074 | 0.148 | 0.074 |
| +Chubby, -Young | 0.149 | 0.005 | 0.363 | 0.026 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 0.228 | 0.013 | 0.468 | 0.013 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 0.000 | 0.029 | 0.000 | 0.059 |
| MACRO | 0.090 | 0.035 | 0.240 | 0.094 |
