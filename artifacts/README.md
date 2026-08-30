# Training results

The short deterministic run used seed 1337, 2 layers, embedding size 64, block size 32, batch size 8, and Adam at 3e-4 for 30 steps.

| Step | Train loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.172241 | 4.140426 |
| 10 | 3.847280 | 3.846327 |
| 20 | 3.655485 | 3.739361 |
| 29 | 3.606934 | 3.569038 |

The expected untrained loss is `ln(65) = 4.174387`. A repeated run produced a byte-identical curve with SHA-256 `5f196d7759e99c9cf43ae7f7f08cb90457279917204479a511edc3b28e349019`.

The single-batch diagnostic used 1 layer, embedding size 32, block size 16, batch size 4, and Adam at 3e-3. Its loss decreased from 4.190166 to 0.006978 in 200 steps.
## 300-step stability run

The real-batch stability run used the same configuration for 300 steps. All training and validation losses remained finite, with no NaN, crash, or divergent-loss exit.

| Step | Train loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.172241 | 4.140426 |
| 50 | 3.322208 | 3.298308 |
| 100 | 2.981515 | 3.002825 |
| 150 | 2.845127 | 2.863898 |
| 200 | 2.757384 | 2.809273 |
| 250 | 2.622646 | 2.805112 |
| 299 | 2.524854 | 2.673984 |

Per-step loss remains noisy, but the 50-step training means decrease across every window: 3.665057, 3.145883, 2.919466, 2.780600, 2.720510, and 2.647938. The first 30-step mean is 3.808987 and the final 30-step mean is 2.642483. The curve is stored in `long_run_curve.csv` with SHA-256 `d2db37d875a6093be426b05ec3ee872d789b377d6ea12fbab070c8468c07a572`.
