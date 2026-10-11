# CONTRACT.md - interfaces every node codes against (repo root R = /home/josh/repos/loom-fm)

Interpreters: PY=/home/josh/miniconda3/envs/mlddpm/bin/python (torch 2.4.1, pytorch_fid, torchvision); PWPY=/home/josh/miniconda3/envs/annoflow/bin/python (playwright; browsers firefox-1543, chromium-1243 installed). Data read-only: /home/josh/repos/data/MNIST/raw (idx files), /home/josh/repos/data/cifar-10-batches-py.
Line budgets (wc -l): loom.py <= 240, train.py <= 230, check.py <= 280, README.md <= 50, site/index.html + site/style.css + site/main.js <= 450 together.

## GPU rules (shared RTX 4080 SUPER, a foreign training uses ~5 GB and ~70% util; never disturb it)
- train.py: `torch.cuda.set_per_process_memory_fraction(0.4)`; batch <= 256; no torch.compile; num_workers <= 2.
- Before training or FID: wait until `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` shows used <= 11000 MiB (poll 60 s, give up after 30 min with exit 3).
- One of our trainings at a time: `fcntl.flock` on R/.gpu.lock for the whole run (blocking). Append own pid to R/.pyramid/pids. Never kill a pid not in that file; never `pkill -f`.
- Nodes run trainings sequentially; a run > 9 min goes to the background (`nohup ... > runs/<name>/log.txt`) and the node waits with `sleep 590` turns (or the Monitor tool), reading only the log tail.

## Paper formulas (LOOM, lam = beta = 0.5, H loops, M shared layers)
- gamma = lam / (H * sqrt(M)); attention branch o = gamma * Attn(RMS(h)); MoE branch: h = h~ + gamma * RMS(MoE(RMS(h~))) (moe_norm applies that outer RMS; without it, h = h~ + gamma * MoE(RMS(h~))). Without res_scale, gamma = 1.
- Re-injection, loop t >= 2 (1-based): g_t = lam / (t * sqrt(M)); h_0^t = (1 - g_t) h_M^{t-1} + g_t x, x = patch embedding (+pos). Loop 1 starts from x.
- Looping residual (EMA): two memories, global (persists over all (t, l)) and local (reset at each loop start): N <- beta N + o, D <- beta D + 1, r = N / D; h~ = h_{l-1} + r_global + r_local (replaces h_{l-1} + o). Without loop_residual: h~ = h_{l-1} + o.
- Routers: per (loop, layer) Linear(dim, E, bias=False) when per_loop_router, else one per layer shared by loops. Softmax over E, top-k, renormalised; Switch load-balance aux loss coefficient 0.01 added to the training loss. Experts shared across loops. Dense (experts = 0): SwiGLU hidden = int(4 * dim * 2 / 3). Expert SwiGLU hidden = int(expert_mult * dim).
- Segmented backprop: loops grouped in segments of K = seg (seg = 0 means one segment). At the end of every segment apply the final layer, compute the flow-matching loss on that output, and detach h before the next segment. Training loss = mean of segment losses. x_pred of forward() = the last segment's output.
- Flow matching (JiT): t ~ sigmoid(N(P_mean, P_std)) with P_mean = -0.8, P_std = 0.8; z = t x + (1 - t) e, e ~ N(0, I); net predicts x; v = (x - z) / max(1 - t, 0.05); loss = mean (v - v_pred)^2. Sampling: Euler from t = 0 to 1 with `nfe` steps, x0 = N(0, I), no guidance (cfg 1.0) for the Pareto front; label dropout 0.1 in training (class index 10 = null). Conditioning c = t_emb + y_emb enters each block through adaLN (6 dim) as in JiT; when per_loop_adaln, each loop owns its adaLN linear.

## loom.py
```
@dataclass
class LoomConfig:
    in_ch: int = 3; img: int = 32; patch: int = 2; dim: int = 256; depth: int = 4; loops: int = 1; heads: int = 4
    experts: int = 0; topk: int = 2; expert_mult: float = 1.0
    res_scale: bool = False; reinject: bool = False; loop_residual: bool = False; per_loop_router: bool = False; moe_norm: bool = False
    seg: int = 0; lam: float = 0.5; beta: float = 0.5; per_loop_adaln: bool = False; num_classes: int = 10
class LoomNet(nn.Module):
    def __init__(self, cfg: LoomConfig)
    gamma: float                                  # the residual scale in use (1.0 when not res_scale)
    def forward(self, z, t, y, stats=False)       # z [B,C,img,img], t [B] in (0,1), y [B] long (num_classes = null)
        -> x_pred [B,C,img,img]  or (x_pred, stats_dict) when stats=True
    def loss(self, x, y) -> (loss scalar incl. aux, info dict with 'fm', 'aux', 'seg_losses': list of scalars one per segment)
    def n_params(self) -> int                    # total trainable
    def n_active(self) -> int                    # parameters touched per token per forward (shared block once per loop; topk experts)
```
stats_dict keys (all detached, float32): `x` embedding [B,N,dim]; `loop_in` list H of h_0^t; `loop_out` list H of h_M^t (pre-detach); `attn_out` dict {(t,l): o} and `resid` dict {(t,l): the residual added to h_{l-1}} with t, l 0-based; `rms` list H of float (RMS of loop_out); `loads` tensor [H, depth, E] mean router probability per expert (zeros when dense).
Unit checks rely on: gamma formula; loop_in[t] == (1-g)*loop_out[t-1] + g*x for t >= 1 when reinject (== loop_out[t-1] otherwise); resid recomputed from attn_out by the EMA formula (allclose 1e-4) when loop_residual, else resid == attn_out; n_params(loops=4, per_loop_router) - n_params(loops=1, per_loop_router) == 3 * depth * dim * experts; with seg=1, loops=2: grad of seg_losses[1] w.r.t. loop-0 router weights is None or zero, grad of seg_losses[0] w.r.t. them is nonzero; shapes above.

## train.py (PY train.py <subcommand> ...)
- `run --dataset {mnist,cifar} --name NAME --tag TAG [config flags: --dim --depth --loops --heads --experts --topk --expert-mult --res-scale --reinject --loop-residual --per-loop-router --moe-norm --seg --lam --beta --per-loop-adaln --loom (sets the five LOOM flags + seg 3)] --steps N --batch 128 --lr 3e-4 --fid-n 5000 --nfe 50 --seed 0 [--eval-every 0]`
  AdamW(0.9, 0.95), wd 0.1 (not on biases/norms), grad clip 1, warmup 5% then constant, EMA 0.999 used for sampling, bf16 autocast. MNIST: 28 -> zero-pad to 32, 1 channel; CIFAR: random horizontal flip. Images in [-1, 1].
  Writes runs/NAME/{samples.npz (uint8 [fid_n, img, img, C], key 'x'), grid.png (8x8 samples, nearest upscale x4), log.txt, ckpt.pt (EMA weights)} and appends one line to R/results.jsonl:
  `{"name","dataset","tag","cfg":{LoomConfig fields},"params","active_params","steps","batch","lr","seed","nfe","fid_n","fid","loss_final","train_min","it_s","peak_gb","finished": ISO-8601}` (extra keys allowed). Skip (exit 0) if NAME already in results.jsonl unless --force.
  FID: pytorch_fid InceptionV3 pool3, samples and reference at 299 bilinear, grey replicated to 3 channels, against R/data/fid_stats_<dataset>.npz (keys mu, sigma).
- `queue FILE.json` : list of argv lists for `run`, executed sequentially (resumable via the skip rule).
- `pareto` : from results.jsonl rows with tag "sweep", per dataset and family ("dense" when experts == 0 and loops == 1, "loom" otherwise) the Pareto front of (params, fid) (a point is on the front if no other point of the same dataset+family has params <= and fid <=), written to R/pareto.json `{dataset: {family: [rows]}}` and R/site/data.json `{"generated", "datasets": {ds: {"front": {family: [{name, params, active_params, fid, cfg, grid}]}, "all": [rows]}}, "iters": [rows with tag starting "iter"]}`; copies each front row's grid.png to site/img/<name>.png.

## check.py (PY check.py <cmd>; exit 0 pass, 1 fail with one-line reasons, 2 usage)
- `selfcheck` stub exercises (gates node only). `stats` writes data/fid_stats_{mnist,cifar}.npz.
- `unit` the loom.py checks above plus line budgets (loom.py, train.py when present).
- `record [NAME|last|--since ISO]` validate rows (schema above, finite fid, params == LoomNet(cfg).n_params(), samples.npz shape and mtime after row's start, grid.png present); recompute FID from samples.npz with its own pytorch_fid call and require |fid - row.fid| < 0.5.
- `smoke` runs `train.py run` for mnist and cifar (steps 60, fid-n 200, batch 32, loom, loops 2, dim 64, depth 2, experts 4, tag smoke, names smoke-<ds>, --force) then `record` on both; < 6 min.
- `iter K` : rows with tag iterK exist, each passes `record`, and R/NOTES.md has a `## iter K` section with lines starting `hypothesis:`, `result:`, `next:`.
- `pareto` : recompute the fronts from results.jsonl and compare with pareto.json and site/data.json; >= 4 dense and >= 4 loom rows with tag sweep per dataset; every front row's site/img/<name>.png exists and is newer than its row; every sweep row passes `record`.
- `readme` : every fenced command line in README.md starting `python`/`PY` names an existing subcommand. `review` : R/.pyramid/review.md has headings `## blockers`, `## majors`, `## minors`.
- `site` (re-execs under PWPY) : serve R/site on 127.0.0.1:8781; for browser in (firefox, chromium) and viewport in (390x844, 1440x900): no console errors or page errors, document.scrollWidth <= clientWidth, every img naturalWidth > 0, no request leaves 127.0.0.1, both dataset charts exist (`svg[data-chart=mnist]`, `svg[data-chart=cifar]`) with one `circle` per front row of data.json, every front row's fid appears in the page text to 1 decimal, design lint (<= 2 font families, <= 1 chromatic hue family, body text contrast >= 4.5, no gradients, no box-shadow blur > 1 px, p measure 45-80 ch at 1440), total transferred bytes <= 1.5 MB, `prefers-reduced-motion` honoured (no running CSS animations when emulated).
