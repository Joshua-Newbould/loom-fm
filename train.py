import argparse, contextlib, copy, fcntl, json, os, pickle, shutil, subprocess, sys, time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent
D = Path('/home/josh/repos/data')
BOOLS = 'res_scale reinject loop_residual per_loop_router moe_norm per_loop_adaln'.split()

def load(ds):
    if ds == 'mnist':
        rd = lambda f, o: np.frombuffer((D / 'MNIST/raw' / f).read_bytes(), np.uint8, offset=o)
        return np.pad(rd('train-images-idx3-ubyte', 16).reshape(-1, 28, 28), ((0, 0), (2, 2), (2, 2)))[:, None], rd('train-labels-idx1-ubyte', 8)
    b = [pickle.load(open(D / f'cifar-10-batches-py/data_batch_{i}', 'rb'), encoding='bytes') for i in range(1, 6)]
    return np.concatenate([d[b'data'] for d in b]).reshape(-1, 3, 32, 32), np.concatenate([d[b'labels'] for d in b])

def wait_gpu():
    import torch
    t0 = time.time()
    torch.cuda.empty_cache()
    while int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).split()[0]) - torch.cuda.memory_reserved() // 2**20 > 11000:
        if time.time() - t0 > 1800: print('GPU stayed above 11000 MiB for 30 min'); sys.exit(3)
        time.sleep(60)

@contextlib.contextmanager
def gpu():
    import torch
    with open(R / '.gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        wait_gpu()
        with open(R / '.pyramid/pids', 'a') as f: f.write(f'{os.getpid()}\n')
        torch.cuda.set_per_process_memory_fraction(0.4)
        yield

@__import__('torch').no_grad()
def sample(net, n, nfe, cfg, bs=250):
    import torch
    out = []
    for i in range(0, n, bs):
        m = min(bs, n - i)
        z, y = torch.randn(m, cfg.in_ch, cfg.img, cfg.img, device='cuda'), (torch.arange(i, i + m, device='cuda') % cfg.num_classes)
        for k in range(nfe):
            t = torch.full((m,), k / nfe, device='cuda')
            with torch.autocast('cuda', torch.bfloat16): x = net(z, t, y).float()
            z = z + (x - z) / (1 - t).clamp(min=0.05).view(-1, 1, 1, 1) / nfe
        out.append(((z.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu())
    return torch.cat(out).numpy()

def fid(x, ds):
    import torch
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    m, f = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval(), []
    with torch.no_grad():
        for i in range(0, len(x), 50):
            b = torch.from_numpy(x[i:i + 50]).cuda().permute(0, 3, 1, 2).float() / 255
            f.append(m(b.repeat(1, 3, 1, 1) if b.shape[1] == 1 else b)[0].flatten(1).cpu().double())
    f = torch.cat(f).numpy()
    ref = np.load(R / 'data' / f'fid_stats_{ds}.npz')
    return float(calculate_frechet_distance(f.mean(0), np.cov(f, rowvar=False), ref['mu'], ref['sigma']))

def run(a):
    out = R / 'runs' / a.name
    rf = R / 'results.jsonl'
    if not a.force and rf.exists() and any(json.loads(l)['name'] == a.name for l in rf.read_text().splitlines() if l.strip()): return
    assert a.batch <= 256
    import torch
    from PIL import Image
    from loom import LoomConfig, LoomNet
    out.mkdir(parents=True, exist_ok=True)
    with gpu():
        t_start = time.time()
        torch.manual_seed(a.seed)
        X, Y = load(a.dataset)
        X, Y = torch.from_numpy(X.copy()).cuda(), torch.from_numpy(Y.astype(np.int64)).cuda()
        kw = {k: getattr(a, k) for k in 'dim depth loops heads experts topk expert_mult seg lam beta'.split() + BOOLS}
        if a.loom: kw.update({k: True for k in 'res_scale reinject loop_residual per_loop_router moe_norm'.split()}, seg=3)
        cfg = LoomConfig(in_ch=X.shape[1], img=32, num_classes=10, **kw)
        net = LoomNet(cfg).cuda()
        ema = copy.deepcopy(net).eval().requires_grad_(False)
        dec = [p for n, p in net.named_parameters() if p.ndim >= 2]
        oth = [p for n, p in net.named_parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([{'params': dec, 'weight_decay': 0.1}, {'params': oth, 'weight_decay': 0.}], lr=a.lr, betas=(0.9, 0.95))
        wu, log, tl, tt = max(1, int(0.05 * a.steps)), open(out / 'log.txt', 'w'), time.time(), time.time()
        for s in range(1, a.steps + 1):
            for g in opt.param_groups: g['lr'] = a.lr * min(1, s / wu)
            i = torch.randint(0, len(X), (a.batch,), device='cuda')
            x = X[i].float() / 127.5 - 1
            if a.dataset == 'cifar': x = torch.where(torch.rand(a.batch, 1, 1, 1, device='cuda') < .5, x.flip(3), x)
            with torch.autocast('cuda', torch.bfloat16): loss, info = net.loss(x, Y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad(): torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), 1 - 0.999)
            if s % 100 == 0 or s == a.steps:
                msg = f'step {s} loss {loss.item():.4f} it/s {100 / (time.time() - tl):.2f}'
                if a.eval_every and s % a.eval_every == 0 and s < a.steps: msg += f' fid {fid(sample(ema, 2000, a.nfe, cfg), a.dataset):.2f}'
                print(msg, flush=True); log.write(msg + '\n'); log.flush(); tl = time.time()
        train_min = (time.time() - tt) / 60
        it_s, loss_final = a.steps / (train_min * 60), float(loss.item())
        wait_gpu()
        xs = sample(ema, a.fid_n, a.nfe, cfg)
        fd = fid(xs, a.dataset)
        np.savez(out / 'samples.npz', x=xs)
        g = xs[:64].reshape(8, 8, 32, 32, -1).transpose(0, 2, 1, 3, 4).reshape(256, 256, -1).repeat(4, 0).repeat(4, 1)
        Image.fromarray(g[..., 0] if g.shape[-1] == 1 else g).save(out / 'grid.png')
        torch.save(ema.state_dict(), out / 'ckpt.pt')
        row = dict(name=a.name, dataset=a.dataset, tag=a.tag, cfg=asdict(cfg), params=net.n_params(), active_params=net.n_active(), steps=a.steps,
                   batch=a.batch, lr=a.lr, seed=a.seed, nfe=a.nfe, fid_n=a.fid_n, fid=fd, loss_final=loss_final,
                   train_min=train_min, it_s=it_s, peak_gb=torch.cuda.max_memory_allocated() / 2**30, finished=datetime.now().isoformat())
        with open(rf, 'a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row))

def queue(a):
    for argv in json.load(open(a.file)):
        subprocess.run([sys.executable, __file__, 'run'] + [str(v) for v in argv], check=True)

def pareto(a):
    rs = [json.loads(l) for l in (R / 'results.jsonl').read_text().splitlines() if l.strip()]
    sw = [r for r in rs if r['tag'] == 'sweep']
    fam = lambda r: 'loom' if r['cfg']['experts'] else ('dense' if r['cfg']['loops'] == 1 else 'looped')
    P, S = {}, {}
    (R / 'site/img').mkdir(parents=True, exist_ok=True)
    for ds in ('mnist', 'cifar'):
        P[ds], S[ds] = {}, {'front': {}, 'all': [r for r in sw if r['dataset'] == ds]}
        for f in ('dense', 'looped', 'loom'):
            v = [r for r in sw if r['dataset'] == ds and fam(r) == f]
            fr = sorted([r for r in v if not any(o['params'] <= r['params'] and o['fid'] <= r['fid'] and (o['params'], o['fid']) != (r['params'], r['fid']) for o in v)], key=lambda r: r['params'])
            P[ds][f] = fr
            S[ds]['front'][f] = [dict({k: r[k] for k in ('name', 'params', 'active_params', 'fid', 'cfg')}, grid=f'img/{r["name"]}.png') for r in fr]
            for r in fr: shutil.copy(R / 'runs' / r['name'] / 'grid.png', R / 'site/img' / f'{r["name"]}.png')
    (R / 'pareto.json').write_text(json.dumps(P, indent=1))
    (R / 'site/data.json').write_text(json.dumps({'generated': datetime.now().isoformat(), 'datasets': S, 'iters': [r for r in rs if r['tag'].startswith('iter')]}))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--dataset', choices=['mnist', 'cifar'], required=True); r.add_argument('--name', required=True); r.add_argument('--tag', required=True)
    for k, v in dict(dim=256, depth=4, loops=1, heads=4, experts=0, topk=2, seg=0).items(): r.add_argument('--' + k, type=int, default=v)
    for k, v in dict(expert_mult=1.0, lam=.5, beta=.5, lr=3e-4).items(): r.add_argument('--' + k.replace('_', '-'), type=float, default=v, dest=k)
    for k in BOOLS + ['loom', 'force']: r.add_argument('--' + k.replace('_', '-'), action='store_true', dest=k)
    for k, v in dict(steps=10000, batch=128, fid_n=5000, nfe=50, seed=0, eval_every=0).items(): r.add_argument('--' + k.replace('_', '-'), type=int, default=v, dest=k)
    sp.add_parser('queue').add_argument('file')
    sp.add_parser('pareto')
    a = p.parse_args()
    {'run': run, 'queue': queue, 'pareto': pareto}[a.cmd](a)
