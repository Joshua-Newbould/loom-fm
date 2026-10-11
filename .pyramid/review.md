# review of loom.py and train.py (diff-scoped, new files)

Checked and found correct: gamma = lam/(H sqrt(M)); g_t = lam/(t sqrt(M)) with 0-based i giving t = i+1 and reinjection applied after the previous detach; EMA memories (global persists and is detached at segment ends, local reset per loop, N <- beta N + o, D <- beta D + 1, r = N/D); router softmax, top-k, renormalise, Switch aux E*sum(P*f) with f normalised by topk, coefficient 0.01, router run in fp32; segment ends apply the shared final layer and detach h, loss is the mean over segments, forward() returns the last segment; n_active and n_params identities; flow-matching loss (t = sigmoid(N(-0.8, 0.8)), z = t x + (1-t) e, v target and prediction divided by max(1-t, 0.05)); label dropout 0.1 with null index 10; Euler sampler uses the EMA net in eval mode, balanced labels, cfg 1.0; FID uses pytorch_fid defaults (uint8 -> [0,1] -> bilinear 299, normalised inside) on EMA samples against stored stats, training data is the train split only, so no leakage.

## blockers
none.

## majors
- train.py:21-27, 96-97: the 11000 MiB wait runs before the flock and never again, so a run that queued behind the lock holder starts (and runs FID) without re-checking foreign usage; fix: move the wait loop inside the flock and call it again before `sample`/`fid`.
- train.py:137: `--batch` is not capped at 256 and `sample` uses bs=500 (line 31); fix: `assert a.batch <= 256` and use bs=250 in sample to respect the contract and the 0.4 memory fraction.

## minors
- train.py:89: EMA decay is a constant 0.999 from step 0, so runs under about 5k steps keep a visible share of the random init (0.999^s) and bias FID against short/small runs; fix: decay = min(0.999, (1+s)/(10+s)).
- train.py:92-95: `eval_every` FID time is included in train_min and it_s, and sampling consumes the global RNG so training batches differ from an eval_every=0 run; fix: subtract eval time and use a dedicated generator in `sample`.
- train.py:95: loss_final is the single last-batch loss including the aux term; fix: average the last 100 `info['fm']` values.
- train.py:40-41: with the 0.05 clamp the last Euler step undershoots (0.4 of the way instead of 1), leaving about 0.012 std of noise in the output (as JiT does); fix: take the final step as `z = x` or ignore (negligible).
- loom.py:139-140: at the first layer of loop 0 global and local memories both equal o so r = 2 o, per contract but a larger first step than the dense residual; fix: none needed, note in the paper write-up.
- loom.py:94-95: adaLN and final layers are init at std 1e-3 rather than zero (deliberate, for router gradients); fix: none, mention as a deviation from JiT.
- loom.py:113-115: nonzero per expert forces E host syncs per layer per loop; fix: only if throughput matters, sort tokens by expert once per layer.
- loom.py:85, 109: `experts < topk` raises in `topk`; fix: assert topk <= experts in LoomConfig use.
- train.py:109-110: queue aborts on the first failed run (including exit 3 GPU timeout); fix: acceptable, resumable by the skip rule.
