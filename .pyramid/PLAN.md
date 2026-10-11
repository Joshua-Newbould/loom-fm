# PLAN.md - LOOM for pixel-space flow matching (MNIST, CIFAR-10): Pareto front of parameters vs FID, plus a site

run_id: 20261011-loom
tier: build - a multi-day research loop (looped MoE recipe ported to JiT-style flow matching), a parameter sweep and a taste-judged site
classification: mixed - sequential shared layer (gates, model, trainer, review), then a research loop (repeat 10, stops when three consecutive iterations fail to improve the best FID at matched params by 3%), then the sweep and the site
routing basis: routing.md derived 2026-10-11; observed.md 115 entries; overrides: site on opus high (standing preference for taste-judged work); model on sonnet high (maths); explore-pick over loop and sweep

## Goals
- Port LOOM (residual scaling, embedding re-injection, looping residual EMA, per-loop routers, MoE output norm, segmented backprop) to a JiT-style x-prediction flow-matching transformer on MNIST (28->32, 1ch) and CIFAR-10 (32, 3ch), pixel space, patch 2.
- Every LOOM component is a flag so the dense baseline, looped-dense and looped-MoE variants come from one model file; the research loop tunes the recipe for this setting (candidates are seeds, not boundaries).
- A Pareto front per dataset of total parameters against FID (10k samples, Euler 50 NFE, no guidance) at a fixed training budget; dense ladder against the best LOOM ladder.
- A minimal, artistic site (static, no build step, offline) that works in Firefox and Chromium on phone and desktop widths.

## Constraints
- Shared RTX 4080 SUPER: one foreign process already uses 4.7 GB and ~70% util. Rules: at most one of our trainings at a time (lock file), `set_per_process_memory_fraction(0.4)`, batch <= 256, start only when free memory >= 5 GB (wait up to 30 min), never kill a process we did not start (`.pyramid/pids`), no torch.compile.
- Python: /home/josh/miniconda3/envs/mlddpm/bin/python (torch 2.4.1, pytorch-fid); browser gate under /home/josh/miniconda3/envs/annoflow/bin/python (playwright, firefox-1543). Data read-only from /home/josh/repos/data.
- Line budgets: loom.py <= 240, train.py <= 230, check.py <= 280, site (html+css+js) <= 450, README <= 50. One check script, no tests beyond contract clauses.
- Quick-screen budget pinned in loop iteration 1 from measured throughput: CIFAR ~10 GPU-min, MNIST ~4 GPU-min, FID on 5k samples; sweep budget 3x quick, FID on 10k.

## Success criteria (each one machine-checkable)
- [ ] model formulas (gamma, g_t re-injection, EMA looping residual, segmented detach, router param deltas, stats shapes) - `check.py unit`
- [ ] every results.jsonl record valid, samples fresh, FID recomputed from samples.npz within 0.5 - `check.py record`
- [ ] end-to-end smoke on both datasets - `check.py smoke`
- [ ] pareto.json recomputed from results.jsonl (tag sweep), >= 4 dense and >= 4 LOOM points per dataset, grids present, site/data.json consistent - `check.py pareto`
- [ ] site: no console errors, no horizontal overflow, images loaded, numbers match data.json, design lint, Firefox + Chromium at 390 and 1440 px - `check.py site`
- [ ] README commands exist - `check.py readme`; review report has blockers/majors/minors - `check.py review`
- research outcome (reported, not gated): LOOM front dominates the dense front at >= 2 budgets per dataset

## Stages
| run | nodes | expected USD | note |
|---|---|---|---|
| 1 | gates, model, train, review, loop(x10), sweep, site | nodes ~5.2, orchestrator ~2.5 | one run; stop at 1.5x = 11.5 |
| 2 | loop2 (x3, CIFAR at the sweep budget) | nodes ~1.5 | extension after the quick-budget recipe lost at 3x budget; unforeseen by the plan, flagged to the user at 16:10 with the extension as the default |
| 3 | sweep2 (looped-dense ladders), site2 (opus) | nodes ~1.5 | third family on both datasets, site rewritten for the new headline |

| id | task_type | agent | model | effort | depends_on | explore | gate |
|---|---|---|---|---|---|---|---|
| gates | implement | implementer | sonnet | medium | - | no | `check.py selfcheck` |
| model | implement | implementer | sonnet | high | gates | no | `check.py unit` |
| train | implement | implementer | sonnet | medium | model | no | `check.py smoke` |
| review | review | implementer | sonnet | high | train | no | `check.py review` |
| loop | research_iter | implementer | per explore-pick | medium | review | pick | `check.py iter` |
| sweep | implement | implementer | per explore-pick | low | loop | pick | `check.py pareto && check.py readme` |
| site | write | implementer | opus | high | sweep | no | `check.py site` |

Node specs (goal, steps, contract, read, files, gate) are in the Workflow call; the run's journal.jsonl keeps them.

## Stop conditions
Gate still failing after escalation; unforeseen review blocker; contract problem that changes the plan; cost past 11.5 USD. Otherwise the run continues to the end.

## Approval
- [x] standing approval from the user's request (2026-10-11): "See this to the end: it may take several days." Session is autonomous; residual choices listed above as assumptions.

## Run log
| run | started | passed/total | re-runs | escalations | nodes USD | orchestrator USD | wall clock | notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-11T13:30+01:00 | 7/7 | 0 | 0 | 5.85 | pending (recorded on the last run) | 5.6 h | transcript wf_ac05b872-dbc; loop stopped at iter 7 by its rule; sweep 24 runs; research outcome negative at 3x budget (LOOM behind dense on both datasets), so run 2 extends the loop at the sweep budget |
| 2 | 2026-10-11T19:15+01:00 | 1/1 | 0 | 0 | 1.57 | pending | 2.5 h | transcript wf_cef52842-c40; loop2 iters 8-10 at the sweep budget: looped dense (no experts, loops 2-3) beats dense d344 on CIFAR on 5/5 runs (53.8 to 58.0 vs 62.9), paper LOOM ties (61.7, 63.6); family rule extended to dense/looped/loom and six runs promoted to the sweep |
