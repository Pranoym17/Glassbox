# Training results

The canonical fixed-seed run used seed 1337, 2 layers, embedding size 64, block size 32, batch size 8, and Adam at 3e-4 for 30 steps.

| Step | Train loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.172241 | 4.140426 |
| 10 | 3.847280 | 3.846327 |
| 20 | 3.655485 | 3.739361 |
| 29 | 3.606934 | 3.569038 |

The expected untrained loss is `ln(65) = 4.174387`. A repeated run produced a byte-identical curve with SHA-256 `5f196d7759e99c9cf43ae7f7f08cb90457279917204479a511edc3b28e349019`.

The single-batch diagnostic used 1 layer, embedding size 32, block size 16, batch size 4, and Adam at 3e-3. Its loss decreased from 4.190166 to 0.006978 in 200 steps.
