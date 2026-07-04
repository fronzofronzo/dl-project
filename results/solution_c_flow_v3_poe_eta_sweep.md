# Solution C — PoE test η-sensitivity (flow_v3_poe)

composition: flow N=8 T=1.0, weighting=none. SENSITIVITY table (η was frozen on train, not selected here).

| η | weighting | R@1 | R@5 | R@10 | 0.5·R@1+0.5·R@5 |
|---|---|---|---|---|---|
| 0.0 | none | 0.091 | 0.282 | 0.411 | 0.1862 |
| 0.003 | none | 0.092 | 0.286 | 0.420 | 0.1886 |
| 0.01 | none | 0.091 | 0.290 | 0.424 | 0.1906 |
| 0.03 | none | 0.088 | 0.281 | 0.422 | 0.1845 |
| 0.1 | none | 0.082 | 0.256 | 0.372 | 0.1688 |
| 0.3 | none | 0.073 | 0.217 | 0.332 | 0.1450 |
| 1.0 | none | 0.065 | 0.179 | 0.278 | 0.1221 |
| 3.0 | none | 0.051 | 0.149 | 0.205 | 0.1004 |
