# LOOM-FM

LOOM (looped mixture-of-experts, "looping beyond twice") applied to JiT-style
x-prediction flow matching on MNIST and CIFAR-10. A small block stack (depth 4)
is looped H times, with MoE FFNs, per-loop routers, residual scaling
(lam = beta = 0.5), input re-injection, loop residual and a moe-norm.

## Recipe found (NOTES.md)
- Segment-wise detach (`--seg 1`) was the big win: -16% FID at loops 3 on CIFAR.
- CIFAR: loops 3, 8 experts, top-1. MNIST: loops 2, 8 experts, top-2.
- Not helpful: per-loop adaLN, lam 1.0/0.25, beta 0.9; loops 4 OOMs under the 0.4 memory cap.

## Pareto fronts (FID, lower is better; sweep = 3x quick steps, batch 64, 10k samples, 50 NFE, 1 seed)
| dataset | family | params -> FID |
|---|---|---|
| CIFAR-10 | dense | 0.33M 107.5, 1.28M 85.2, 2.84M 73.6, 5.01M 71.9, 9.00M 62.9, 14.7M 60.1 |
| CIFAR-10 | LOOM | 0.60M 124.2, 1.32M 105.5, 2.33M 94.0, 5.21M 79.4, 9.22M 72.3, 14.4M 64.5 |
| MNIST | dense | 0.33M 109.8, 1.27M 85.4, 2.83M 76.1, 5.00M 74.8, 14.7M 73.4 |
| MNIST | LOOM | 0.59M 161.4, 1.32M 115.4, 2.33M 86.2, 5.20M 79.6, 9.21M 74.5 |

At this budget LOOM does not dominate dense at any parameter point on either dataset.
(The 3400-step iteration runs showed a CIFAR gain; it does not survive the longer, parameter-laddered sweep.)

## Reproduce
    PY=/home/josh/miniconda3/envs/mlddpm/bin/python
    $PY train.py queue sweep.json
    $PY train.py pareto && $PY check.py pareto
