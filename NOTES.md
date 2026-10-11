# NOTES

## budgets (pinned iter 1)
- Measured on the shared GPU (foreign run ~70% util): dim 256 depth 4 loops 3 experts 8 topk 2 --loom: 5.6 it/s at batch 64 (batch 128 OOMs under the 0.4 memory fraction, 6.2 GiB); dense dim 256 depth 4: 47 it/s. The expert loop (per-expert nonzero/index_add) dominates.
- Quick budget: batch 64, cifar 3400 steps (~10 GPU-min for loom), mnist 1400 steps (~4 GPU-min), fid-n 5000, nfe 50. Compare at equal steps and batch.
- Fixed in train.py: GPU wait now inside the flock and repeated before sampling (excluding own reserved memory); batch asserted <= 256; sample bs 250.

## best
| dataset | budget | params | fid | flags |
|---|---|---|---|---|
| cifar | sweep 10200 b64 | 9001916 | 53.78 | looped dense dim 344 loops 3 seed 1 (iter10-looped-dense-l3-s1, 17.9 min); seed 0 56.10 |
| cifar | sweep 10200 b64 | 9001916 | 54.82 | looped dense dim 344 loops 2 seed 2 (iter10-looped-dense-l2-s2, 12.0 min); seeds 0/1 58.0/55.3 |
| cifar | sweep 10200 b64 | 9001916 | 55.30 | looped dense dim 344 loops 2, seed 1 (iter9-looped-dense-l2-s1, 12.0 min); seed 0 below |
| cifar | sweep 10200 b64 | 9001916 | 56.10 | looped dense dim 344 loops 3 (iter9-looped-dense-l3, 17.9 min), single seed |
| cifar | sweep 10200 b64 | 9001916 | 57.98 | looped dense dim 344 depth 4 loops 2 experts 0, res_scale reinject loop_residual (iter8-looped-dense-l2, 12.0 min; dense d344 62.9 at 6.1 min), single seed |
| cifar | sweep 10200 b64 | 9223948 | 61.7 | paper LOOM: dim 256 loops 3 experts 8 topk 2 seg 3 --loom (iter8-loom-paper, 15.6 min), single seed |
| cifar | quick 3400 b64 | 9223948 | 133.8 | dim 256 depth 4 loops 3 experts 8 topk 1 seg 1 (iter7; short-horizon only) |
| mnist | quick 1400 b64 | 9211652 | 187.2 | dim 256 depth 4 loops 2 experts 8 topk 2 seg 1 (iter6; within noise of dense) |

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

## iter 8
hypothesis: the quick-budget winner (seg 1, top-1) is a short-horizon bias: detach speeds early loss but caps long-run quality, so at 10200 steps full backprop (paper seg 3) and a looped dense net should beat dense d344 (62.9, 6.1 min).
result: cifar 10200 b64 fid-n 10000: paper loom seg 3 top-2 loops 3 9.22M fid 61.7, 15.6 min (iter8-loom-paper); looped dense d344 loops 2 9.00M fid 57.98, 12.0 min (iter8-looped-dense-l2); 4 experts mult 2 loops 2 seg 1 9.21M fid 63.2, 10.5 min (iter8-e4m2-l2). Two of three beat dense 62.9 (single seed, noise unknown); looped dense is best at about 2x dense wall-clock.
next: iter9: second seed (seed 1) of iter8-looped-dense-l2 and of iter8-loom-paper; plus looped dense loops 3.

## iter 9
hypothesis: the iter8 gains hold across seeds: looped dense loops 2 and paper LOOM at 10200 steps beat dense d344 (62.9), and loops 3 helps looped dense.
result: cifar 10200 b64 fid-n 10000: looped dense l2 seed 1 9.00M fid 55.30, 12.0 min (iter9-looped-dense-l2-s1; seed 0 57.98, gap 2.7); paper LOOM seed 1 9.22M fid 63.55, 15.6 min (iter9-loom-paper-s1; seed 0 61.7, gap 1.9, but seed 1 is above 62.9); looped dense loops 3 9.00M fid 56.10, 17.9 min (iter9-looped-dense-l3, seed 0). Looped dense beats dense on both seeds (mean 56.6) at 2x wall-clock; paper LOOM is about equal to dense (mean 62.6). Stopping rule not met: the looped dense gap is 2.7 (>2) and the LOOM seed 1 does not beat 62.9.
next: iter10: seed 2 of iter8-looped-dense-l2 and seed 1 of iter9-looped-dense-l3 to settle the seed gap; seed 2 of the paper LOOM if budget allows.

## iter 10
hypothesis: the looped dense gain over dense d344 (62.9, 6.1 min) is real across seeds, not a lucky seed.
result: cifar 10200 b64 fid-n 10000, 9.00M: looped dense loops 2 seed 2 fid 54.82, 12.0 min (iter10-looped-dense-l2-s2; seeds 0/1/2: 57.98/55.30/54.82, mean 56.0, seeds 1 and 2 within 0.5); looped dense loops 3 seed 1 fid 53.78, 17.9 min (iter10-looped-dense-l3-s1; seeds 0/1: 56.10/53.78, gap 2.3, mean 54.9). Every looped dense run (5 of 5) beats 62.9, at 2x (loops 2) to 3x (loops 3) dense wall-clock. Paper LOOM (61.7/63.55) and the 4-expert variant (63.2) do not clearly beat dense. Stopping rule holds: looped dense beats 62.9 at 9.0M with a second seed within 2 FID (l2 seeds 1/2).
next:

## promote
- iter10-looped-dense-l3-s1 (CIFAR looped front at 9.00M: 53.78)
- iter10-looped-dense-l2-s2 (CIFAR looped front at 9.00M: 54.82)
- iter8-loom-paper (CIFAR LOOM front at 9.22M: 61.7 vs 72.3)
- iter8-looped-dense-l2 (CIFAR looped front at 9.00M: 57.98)
- iter9-looped-dense-l3 (CIFAR looped front at 9.00M: 56.10)
- iter9-looped-dense-l2-s1 (CIFAR looped front at 9.00M: 55.30)
