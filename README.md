# LOOM-FM

LOOM (looped mixture-of-experts, "looping beyond twice") applied to JiT-style
x-prediction flow matching on MNIST and CIFAR-10. A depth-4 block stack is looped H times.

## Recipe found (NOTES.md)
- Looped dense wins on CIFAR: no experts, loops 2, residual scaling, input re-injection and
  loop residual (`--res-scale --reinject --loop-residual`, depth 4). At 9.0M params it reaches FID 57.98
  (seeds 55 to 58; loops 3 reaches 53.8 to 56.1) versus 62.9 for dense and 61.7 for paper LOOM
  (dim 256, loops 3, 8 experts top-2, seg 3, 9.2M).
- Paper LOOM ties looped dense only at the largest budgets; MoE alone (loops 1) is worse than dense.
- On MNIST looped dense does not beat dense at any budget.
- Not helpful: per-loop adaLN, lam 1.0/0.25, beta 0.9; loops 4 OOMs under the 0.4 memory cap.

## Pareto fronts (FID, lower is better; sweep = 10200 steps CIFAR, 4200 MNIST, batch 64, 10k samples, 50 NFE)
| dataset | family | params and FID |
|---|---|---|
| CIFAR-10 | dense | 0.33M 107.5, 1.28M 85.2, 2.84M 73.6, 5.01M 71.9, 9.00M 62.9, 14.68M 60.1 |
| CIFAR-10 | looped dense | 0.33M 94.5, 1.28M 77.4, 2.84M 76.0, 5.01M 64.5, 9.00M 53.8, 14.68M 53.5 |
| CIFAR-10 | LOOM | 0.60M 124.2, 1.32M 105.5, 2.33M 94.0, 5.21M 79.4, 9.22M 61.7 |
| MNIST | dense | 0.33M 109.8, 1.27M 85.4, 2.83M 76.1, 5.00M 74.8, 14.68M 73.4 |
| MNIST | looped dense | 0.33M 117.8, 1.27M 87.1, 2.83M 78.9, 5.00M 76.6, 9.00M 75.2 |
| MNIST | LOOM | 0.59M 161.4, 1.32M 115.4, 2.33M 86.2, 5.20M 79.6, 9.21M 74.5 |

Looped dense beats dense at 5 of 6 CIFAR budgets (not at 2.84M) and at no MNIST budget.
Single seed per point except the 9.0M looped CIFAR points.

## Reproduce
    PY=/home/josh/miniconda3/envs/mlddpm/bin/python
    $PY train.py queue sweep.json
    $PY train.py queue sweep2.json
    $PY train.py pareto && $PY check.py pareto
