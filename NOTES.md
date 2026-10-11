# NOTES

## budgets (pinned iter 1)
- Measured on the shared GPU (foreign run ~70% util): dim 256 depth 4 loops 3 experts 8 topk 2 --loom: 5.6 it/s at batch 64 (batch 128 OOMs under the 0.4 memory fraction, 6.2 GiB); dense dim 256 depth 4: 47 it/s. The expert loop (per-expert nonzero/index_add) dominates.
- Quick budget: batch 64, cifar 3400 steps (~10 GPU-min for loom), mnist 1400 steps (~4 GPU-min), fid-n 5000, nfe 50. Compare at equal steps and batch.
- Fixed in train.py: GPU wait now inside the flock and repeated before sampling (excluding own reserved memory); batch asserted <= 256; sample bs 250.

## best
| dataset | params | fid | flags |
|---|---|---|---|
| cifar | 9223948 | 133.8 | dim 256 depth 4 loops 3 experts 8 topk 1, all 5 components, seg 1 (explicit flags, not --loom), 3400 steps b64 (iter7; 2.4% better than topk 2 137.05, 20% better than dense 167.8) |
| mnist | 9211652 | 187.2 | dim 256 depth 4 loops 2 experts 8 topk 2, all 5 components, seg 1, 1400 steps b64 (iter6; 0.9% better than dense 188.8, within noise) |

## iter 1
hypothesis: full LOOM (3 loops, 8 experts top-2, all five components, seg 3) beats a dense net of matched total params (~9.0-9.2M) at equal steps and batch.
result: cifar dense dim344 9.00M fid 167.8 (iter1-dense-cifar); cifar loom 9.22M fid 164.0 (iter1-loom-cifar), active 12.9M, 2.3% better; mnist dense 9.00M fid 188.8 (iter1-dense-mnist); mnist loom 9.22M fid 202.7 (iter1-loom-mnist), worse. FIDs are high because budgets are tiny (3400/1400 steps). LOOM costs 2-3x the wall time of dense (expert loop, 3 loops).
next: ablate on cifar at equal steps: loops 1 with experts 8 (is looping or MoE giving the gain), then loops 2 vs 3, then drop loop_residual.

## iter 2
hypothesis: the cifar gain over dense comes from looping (loop_residual among the components), not just MoE.
result: cifar 3400 steps b64, all ~9.2M params: loops1 experts8 --loom fid 176.9 (iter2-l1, 1.8 min); loops2 --loom fid 161.1 (iter2-l2, 3.5 min); loops3 --loom without loop_residual fid 172.3 (iter2-l3-noresid, 5.1 min; --loom forces loop_residual on, so run with explicit flags). Versus iter1: loops3 full 164.0, dense 167.8. MoE alone (loops1) is worse than dense; looping helps; loop_residual is worth ~8 FID at loops 3; loops 2 is slightly better and cheaper than 3 (single seed, noise unknown).
next: iter3 on cifar: loops 2 with seg 0 (full backprop) vs seg 2, and loops 4 --loom.

## iter 3
hypothesis: at loops 2 with full LOOM, detaching segments (seg 2) vs full backprop (seg 0) changes FID; and loops 4 beats loops 2.
result: seg is counted in loops, so at loops 2 seg 0, 2 and 3 are all full backprop: iter3-l2-seg0 and iter3-l2-seg2 (9.22M) gave fid 161.07, bit-identical to iter2-l2 (degenerate test, no information). Loops 4 (9.22M) OOMs under the 0.4 memory fraction (6.2 GiB) at batch 64 for --loom (seg 3), seg 2 and seg 1 alike; no loops-4 result. Best unchanged (cifar 161.1). Loops 4 needs memory-lean experts or batch 32 (not equal-batch).
next: test seg effect at loops 3 (seg 1 vs seg 3 = 164.0) on cifar, then per_loop_adaln at loops 2, then lam/beta.

## iter 4
hypothesis: segment-wise detach (seg 1) beats full backprop (seg 3) at loops 3, as the paper's truncated backprop suggests.
result: cifar 3400 steps b64. loops3 seg1 (9.22M) fid 137.05 (iter4-l3-seg1; vs seg3 164.0, 16% better; 6.4 min). loops2 seg1 (9.22M) fid 152.83 (iter4-l2-seg1; vs full backprop 161.1). loops3 seg1 + per_loop_adaln (12.38M, not matched) fid 157.76 (iter4-l3-seg1-adaln): worse despite more params. Seg 1 is a large win; keep it. Single seed.
next: iter5 cifar: loops 3 seg 1 with lam/beta variants, then loops 4 seg 1 (memory), then mnist with seg 1.

## iter 5
hypothesis: lam (residual scale) or beta (EMA memory decay) is mis-set for seg 1 at loops 3; mnist also benefits from seg 1.
result: cifar 3400 steps b64, 9.22M: lam 1.0 fid 140.93 (iter5-l3-seg1-lam1; vs 137.05 at lam 0.5, worse); beta 0.9 fid 149.33 (iter5-l3-seg1-beta09, worse). mnist 1400 steps loops3 seg1 (9.22M) fid 227.9 (iter5-mnist-l3-seg1) vs dense 188.8 and seg3 loom 202.7: worse; on mnist looping/MoE does not pay at this budget. Best table unchanged. Single seed.
next: iter6: cifar lam 0.25 and beta 0.25 at loops 3 seg 1; mnist loops 2 seg 1 or dense-ish loops 1 to find what helps mnist; loops 4 seg 1 still OOM-blocked.

## iter 6
hypothesis: lam 0.5 and beta 0.5 are not optimal for seg 1 at loops 3; mnist needs fewer loops.
result: cifar 3400 steps b64, 9.22M, loops3 seg1: lam 0.25 fid 149.97 (iter6-l3-seg1-lam025, worse); beta 0.25 fid 138.92 (iter6-l3-seg1-beta025, vs 137.05, about equal/worse). lam 0.5, beta 0.5 stay. mnist 1400 steps loops2 seg1 (9.21M) fid 187.15 (iter6-mnist-l2-seg1) vs dense 188.8 (0.9%, within noise, not >=3%). Single seed.
next: iter7: cifar loops 3 seg 1 with topk 1 or experts 4 with expert_mult 2; mnist loops 2 seg 1 with experts 4; seed repeat of best if budget allows.

## iter 7
hypothesis: fewer active experts per token (topk 1) or fewer, wider experts (4 x mult 2) help at loops 3 seg 1.
result: cifar 3400 steps b64, 9.22M: topk 1 fid 133.81 (iter7-l3-seg1-topk1, 4.8 min; vs 137.05, 2.4% better, below 3%, best row updated). Cifar experts 4 mult 2 OOMs under the 0.4 memory fraction (6.2 GiB), no result. Mnist loops 2 seg 1 experts 4 mult 2 (9.20M) fid 190.13 (iter7-mnist-l2-seg1-e4m2; vs 187.15 best, worse). Single seed. Iters 5, 6, 7 each improved no best by >= 3%: stopping rule holds.
next:
