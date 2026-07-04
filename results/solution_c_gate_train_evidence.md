# Solution C — gate evidence on TRAIN pseudo-queries (thresh=2)

train split, 200 sources/query, seed 0. The |C| complementarity was first noticed on test tables; this table is the blind confirmation used to freeze the gate.

| query | C | t2 R@1 | flow R@1 | t2 R@5 | flow R@5 |
|---|---|---|---|---|---|
| +Smiling | 1 | 0.045 | 0.060 | 0.150 | 0.175 |
| +Eyeglasses | 1 | 0.100 | 0.130 | 0.310 | 0.345 |
| -Heavy_Makeup | 1 | 0.025 | 0.040 | 0.105 | 0.155 |
| +Male | 1 | 0.065 | 0.070 | 0.185 | 0.215 |
| -Young | 1 | 0.010 | 0.035 | 0.115 | 0.155 |
| +Blond_Hair | 1 | 0.045 | 0.075 | 0.170 | 0.225 |
| +Mustache | 1 | 0.010 | 0.055 | 0.090 | 0.165 |
| +Eyeglasses, +Smiling | 2 | 0.035 | 0.085 | 0.155 | 0.240 |
| +Black_Hair, -Wavy_Hair | 2 | 0.055 | 0.040 | 0.225 | 0.135 |
| -Male, -Mustache | 2 | 0.045 | 0.055 | 0.120 | 0.190 |
| +Chubby, -Young | 2 | 0.015 | 0.030 | 0.100 | 0.130 |
| -Smiling, +Eyeglasses, +Wearing_Hat | 3 | 0.150 | 0.150 | 0.435 | 0.475 |
| +Wearing_Lipstick, -Heavy_Makeup, +Smiling | 3 | 0.005 | 0.025 | 0.020 | 0.060 |

| group | expert | R@1 | R@5 |
|---|---|---|---|
| |C| < 2 | t2 | 0.043 | 0.161 |
| |C| < 2 | flow | 0.066 | 0.205 |
| |C| >= 2 | t2 | 0.051 | 0.176 |
| |C| >= 2 | flow | 0.064 | 0.205 |

**gate thresh=2: NOT CONFIRMED (0.5·R@1+0.5·R@5 — single: t2 0.1018 vs flow 0.1357; multi: t2 0.1133 vs flow 0.1346)**
