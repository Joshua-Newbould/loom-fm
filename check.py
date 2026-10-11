import contextlib, fcntl, json, math, os, pickle, re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('LOOM_ROOT', R))  # results, runs, site, notes (overridable for selfcheck)
PWPY = '/home/josh/miniconda3/envs/annoflow/bin/python'
KEYS = 'name dataset tag cfg params active_params steps batch lr seed nfe fid_n fid loss_final train_min it_s peak_gb finished'.split()
_model = None
def rows():
    f = ROOT / 'results.jsonl'
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
lines = lambda p: len(Path(p).read_text().splitlines())
@contextlib.contextmanager
def gpu():
    t0 = time.time()
    while int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).split()[0]) > 11000:
        if time.time() - t0 > 1800: print('FAIL: GPU stayed above 11000 MiB for 30 min'); sys.exit(3)
        time.sleep(60)
    with open(R / '.gpu.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(R / '.pyramid' / 'pids', 'a') as f:
            f.write(f'{os.getpid()}\n')
        import torch
        torch.cuda.set_per_process_memory_fraction(0.4)
        yield
        fcntl.flock(lk, fcntl.LOCK_UN)
def feats(x):
    global _model
    import torch
    from pytorch_fid.inception import InceptionV3
    _model = _model or InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 50):
            b = torch.from_numpy(x[i:i + 50]).cuda().permute(0, 3, 1, 2).float() / 255
            out.append(_model(b.repeat(1, 3, 1, 1) if b.shape[1] == 1 else b)[0].flatten(1).cpu().double())
    return torch.cat(out).numpy()
def stat(x):
    import numpy as np
    with gpu(): a = feats(x)
    return a.mean(0), np.cov(a, rowvar=False)
def fid(x, ds):
    import numpy as np
    from pytorch_fid.fid_score import calculate_frechet_distance
    mu, sg = stat(x)
    ref = np.load(R / 'data' / f'fid_stats_{ds}.npz')
    return float(calculate_frechet_distance(mu, sg, ref['mu'], ref['sigma']))
def stats(a):
    import numpy as np
    d = Path('/home/josh/repos/data')
    m = np.frombuffer((d / 'MNIST/raw/train-images-idx3-ubyte').read_bytes(), np.uint8, offset=16).reshape(-1, 28, 28)
    c = [pickle.load(open(d / f'cifar-10-batches-py/data_batch_{i}', 'rb'), encoding='bytes')[b'data'] for i in range(1, 6)]
    xs = {'mnist': np.pad(m, ((0, 0), (2, 2), (2, 2)))[..., None],  # zero (black) pad 28 -> 32
          'cifar': np.concatenate(c).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)}
    for k, x in xs.items():
        mu, sg = stat(np.ascontiguousarray(x))
        np.savez(R / 'data' / f'fid_stats_{k}.npz', mu=mu, sigma=sg)
        print(f'stats {k}: n={len(x)} mu{mu.shape} sigma{sg.shape}')
    return []
def unit(a):
    import torch
    sys.path.insert(0, str(ROOT))
    errs = [f'{f} has {lines(ROOT / f)} lines > {n}' for f, n in (('loom.py', 240), ('train.py', 230)) if (ROOT / f).exists() and lines(ROOT / f) > n]
    from loom import LoomConfig, LoomNet
    torch.manual_seed(0)
    M, E, H, lam, beta = 2, 4, 3, .5, .5
    mk = lambda **k: LoomNet(LoomConfig(in_ch=1, img=32, patch=4, dim=32, depth=M, heads=2, experts=E, **k)).eval()
    ok = lambda c, msg: c or errs.append(msg)
    z, t, y = torch.randn(2, 1, 32, 32), torch.rand(2) * .9 + .05, torch.tensor([0, 10])
    for fl in (dict(res_scale=True, reinject=True, loop_residual=True, per_loop_router=True, moe_norm=True), {}):
        n = mk(loops=H, **fl)
        with torch.no_grad():
            o, s = n(z, t, y, stats=True)
        ok(o.shape == z.shape and s['loads'].shape == (H, M, E), 'x_pred / loads shape')
        ok(abs(n.gamma - (lam / (H * M ** .5) if fl else 1.0)) < 1e-6, 'gamma formula')
        for i in range(1, H):
            g = lam / ((i + 1) * M ** .5) if fl else 0
            ok(torch.allclose(s['loop_in'][i], (1 - g) * s['loop_out'][i - 1] + g * s['x'], atol=1e-4), f'loop_in[{i}] re-injection')
        N = Dg = 0
        for i in range(H):
            Nl = Dl = 0
            for l in range(M):
                o_ = s['attn_out'][(i, l)]
                N, Dg, Nl, Dl = beta * N + o_, beta * Dg + 1, beta * Nl + o_, beta * Dl + 1
                ok(torch.allclose(s['resid'][(i, l)], N / Dg + Nl / Dl if fl else o_, atol=1e-4), f'resid[{i},{l}] {"EMA" if fl else "== attn_out"}')
    d = mk(loops=4, per_loop_router=True).n_params() - mk(loops=1, per_loop_router=True).n_params()
    ok(d == 3 * M * 32 * E, f'router param delta {d} != {3 * M * 32 * E}')
    n = mk(loops=2, seg=1, per_loop_router=True).train()
    _, info = n.loss(torch.randn(2, 1, 32, 32).clamp(-1, 1), y)
    rt = [p for k, p in n.named_parameters() if 'router' in k]
    g = [[None if r is None else r.abs().sum().item() for r in torch.autograd.grad(L, rt, retain_graph=True, allow_unused=True)] for L in info['seg_losses']]
    early = [i for i in range(len(rt)) if not g[1][i]]  # routers that seg-1 loss cannot reach = loop 0
    ok(len(rt) == 2 * M and early and all(g[0][i] for i in early) and any(g[1]), 'seg=1 gradient isolation of loop-0 routers')
    return errs
def rec_errs(r):
    import numpy as np
    n = r.get('name', '?')
    if [k for k in KEYS if k not in r]: return [f'{n}: row missing keys {[k for k in KEYS if k not in r]}']
    if not isinstance(r['fid'], (int, float)) or not math.isfinite(r['fid']): return [f'{n}: fid not finite']
    c = r['cfg']
    if not os.environ.get('LOOM_SELFCHECK'):
        sys.path.insert(0, str(R))
        from loom import LoomConfig, LoomNet
        if (p := LoomNet(LoomConfig(**c)).n_params()) != r['params']: return [f'{n}: params {r["params"]} != n_params() {p}']
    s, g = ROOT / 'runs' / n / 'samples.npz', ROOT / 'runs' / n / 'grid.png'
    if not (s.exists() and g.exists()): return [f'{n}: missing {s if not s.exists() else g}']
    if s.stat().st_mtime < datetime.fromisoformat(r['finished']).timestamp() - 60 * r['train_min'] - 2:
        return [f'{n}: samples.npz older than the run start']
    x = np.load(s)['x']
    if x.dtype != np.uint8 or x.shape != (r['fid_n'], c['img'], c['img'], c['in_ch']): return [f'{n}: samples shape/dtype {x.shape} {x.dtype}']
    f = fid(x, r['dataset'])
    return [f'{n}: recomputed fid {f:.2f} vs row {r["fid"]:.2f}'] if abs(f - r['fid']) >= .5 else []
def record(a):
    rs, sel = rows(), (a or ['last'])[0]
    rs = [r for r in rs if r.get('finished', '') >= a[1]] if sel == '--since' else rs[-1:] if sel == 'last' else [r for r in rs if r.get('name') == sel][-1:]
    return [e for r in rs for e in rec_errs(r)] if rs else ['record: no matching rows in results.jsonl']
def smoke(a):
    t0 = time.time()
    if not (ROOT / 'train.py').exists(): return ['smoke: train.py missing']
    for ds in ('mnist', 'cifar'):
        p = subprocess.run([sys.executable, str(ROOT / 'train.py'), 'run', '--dataset', ds, '--name', f'smoke-{ds}', '--tag', 'smoke', '--steps', '60', '--fid-n', '200',
                            '--batch', '32', '--loom', '--loops', '2', '--dim', '64', '--depth', '2', '--experts', '4', '--force'], capture_output=True, text=True, timeout=360)
        if p.returncode: return [f'smoke-{ds}: train.py exit {p.returncode}: {(p.stderr.strip().splitlines() or [""])[-1]}']
    return record(['smoke-mnist']) + record(['smoke-cifar']) + ([f'smoke took {time.time() - t0:.0f}s > 360s'] if time.time() - t0 > 360 else [])
def iter_(a):
    k = a[0]
    rs = [r for r in rows() if r.get('tag') == f'iter{k}']
    notes = ROOT / 'NOTES.md'
    sec = re.search(rf'^## iter {k}\s*$(.*?)(?=^## |\Z)', notes.read_text() if notes.exists() else '', re.S | re.M)
    return ([] if rs else [f'iter {k}: no rows with tag iter{k}']) + [e for r in rs for e in rec_errs(r)] + (
        [f'iter {k} section lacks a "{w}:" line' for w in ('hypothesis', 'result', 'next') if not re.search(rf'^{w}:', sec[1], re.M)] if sec else [f'NOTES.md has no "## iter {k}" section'])
fam = lambda r: 'dense' if r['cfg']['experts'] == 0 and r['cfg']['loops'] == 1 else 'loom'
def pareto(a):
    errs, rs = [], [r for r in rows() if r.get('tag') == 'sweep']
    try:
        pj, sj = json.loads((ROOT / 'pareto.json').read_text()), json.loads((ROOT / 'site/data.json').read_text())['datasets']
    except (OSError, ValueError, KeyError) as e:
        return [f'pareto: cannot read pareto.json / site/data.json ({e.__class__.__name__})']
    for ds in ('mnist', 'cifar'):
        for f in ('dense', 'loom'):
            v = [r for r in rs if r['dataset'] == ds and fam(r) == f]
            fr = [r for r in v if not any(o['params'] <= r['params'] and o['fid'] <= r['fid'] and (o['params'], o['fid']) != (r['params'], r['fid']) for o in v)]
            if len(v) < 4: errs.append(f'{ds}/{f}: {len(v)} sweep rows < 4')
            for src, got in (('pareto.json', pj.get(ds, {}).get(f, [])), ('data.json', sj.get(ds, {}).get('front', {}).get(f, []))):
                if sorted(r['name'] for r in got) != sorted(r['name'] for r in fr):
                    errs.append(f'{ds}/{f}: {src} front differs from recomputed {sorted(r["name"] for r in fr)}')
            for r in fr:
                img = ROOT / 'site/img' / f'{r["name"]}.png'
                if not img.exists() or img.stat().st_mtime < datetime.fromisoformat(r['finished']).timestamp():
                    errs.append(f'{r["name"]}: site/img png missing or older than its row')
    return errs + [e for r in rs for e in rec_errs(r)]
def readme(a):
    p = ROOT / 'README.md'
    if not p.exists(): return ['README.md missing']
    cmds = [l.strip() for b in re.findall(r'```[^\n]*\n(.*?)```', p.read_text(), re.S) for l in b.splitlines()]
    ms = [(l, re.match(r'(?:python|PY)\s+(\S+\.py)\s+(\S+)', l)) for l in cmds]
    return ([f'README.md has {lines(p)} lines > 50'] if lines(p) > 50 else []) + [f'README command not found: {l}' for l, m in ms if m and not (
        (ROOT / m[1]).exists() and re.search(rf'[\'"]{re.escape(m[2])}[\'"]', (ROOT / m[1]).read_text()))]
def review(a):
    p = ROOT / '.pyramid/review.md'
    t = p.read_text() if p.exists() else ''
    return [f'review.md lacks "## {h}"' for h in ('blockers', 'majors', 'minors') if not re.search(rf'^## {h}\s*$', t, re.M)]
JS = r"""A=>{
const P=s=>{const m=(s||'').match(/rgba?\(([^)]+)\)/);if(!m)return null;const v=m[1].split(/[ ,\/]+/).map(parseFloat);return{r:v[0],g:v[1],b:v[2],a:v.length>3?v[3]:1}};
const L=c=>{const f=u=>{u/=255;return u<=.03928?u/12.92:((u+.055)/1.055)**2.4};return .2126*f(c.r)+.7152*f(c.g)+.0722*f(c.b)};
const els=[...document.querySelectorAll('body, body *')],cs=els.map(e=>getComputedStyle(e)),e=[],de=document.documentElement;
if(de.scrollWidth>de.clientWidth)e.push(`horizontal overflow ${de.scrollWidth}>${de.clientWidth}`);
const bad=[...document.images].filter(i=>!(i.complete&&i.naturalWidth>0)).length;if(bad)e.push(bad+' broken img');
for(const d of['mnist','cifar']){const s=document.querySelector(`svg[data-chart=${d}]`),w=(A.data[d]||[]).length,n=s?s.querySelectorAll('circle').length:-1;
 if(n!==w)e.push(`svg[data-chart=${d}] has ${n} circles, want ${w}`);
 for(const f of A.data[d]||[])if(!document.body.textContent.includes(f.toFixed(1)))e.push(`fid ${f.toFixed(1)} not in page text`)}
const fonts=new Set(cs.map(c=>c.fontFamily.split(',')[0].trim().replace(/["']/g,'').toLowerCase()));if(fonts.size>2)e.push('font families '+[...fonts]);
const hues=[];for(const c of cs)for(const k of['color','backgroundColor','borderTopColor','fill','stroke']){const v=P(c[k]);if(!v||v.a===0||(k==='borderTopColor'&&c.borderTopWidth==='0px'))continue;
 const mx=Math.max(v.r,v.g,v.b),mn=Math.min(v.r,v.g,v.b),d=mx-mn;if(d<20||d/mx<.2)continue;
 hues.push((60*(mx===v.r?((v.g-v.b)/d)%6:mx===v.g?(v.b-v.r)/d+2:(v.r-v.g)/d+4)+360)%360)}
if(hues.some(h=>Math.min(Math.abs(h-hues[0]),360-Math.abs(h-hues[0]))>40))e.push('more than one chromatic hue family');
if(cs.some(c=>c.backgroundImage.includes('gradient'))||document.querySelector('linearGradient,radialGradient'))e.push('gradients present');
const blur=Math.max(0,...cs.filter(c=>c.boxShadow!=='none').flatMap(c=>c.boxShadow.replace(/rgba?\([^)]*\)/g,'').split(',').map(s=>parseFloat(s.replace('inset','').trim().split(/\s+/)[2]||0))));
if(blur>1)e.push(`box-shadow blur ${blur}px`);
const bg=x=>{for(;x;x=x.parentElement){const b=P(getComputedStyle(x).backgroundColor);if(b&&b.a>0)return b}return{r:255,g:255,b:255}};
const ct=Math.min(...[document.body,...document.querySelectorAll('p')].map(x=>{const a=L(P(getComputedStyle(x).color)),b=L(bg(x));return(Math.max(a,b)+.05)/(Math.min(a,b)+.05)}));
if(ct<4.5)e.push('text contrast '+ct.toFixed(2));
const cv=document.createElement('canvas').getContext('2d');
if(A.w>=1440)for(const p of document.querySelectorAll('p')){if(p.textContent.length<100)continue;const c=getComputedStyle(p);cv.font=`${c.fontStyle} ${c.fontWeight} ${c.fontSize} ${c.fontFamily}`;
 const ch=(p.clientWidth-parseFloat(c.paddingLeft)-parseFloat(c.paddingRight))/cv.measureText('0').width;if(ch<45||ch>80)e.push('p measure '+ch.toFixed(0)+'ch')}
return e}"""
ANIM = '()=>document.getAnimations().filter(a=>a.playState==="running").length'
def site(a):
    if not sys.executable.startswith(os.path.dirname(PWPY)): os.execv(PWPY, [PWPY, __file__, 'site'])
    import functools, threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from playwright.sync_api import sync_playwright
    sd, errs, data = ROOT / 'site', [], {}
    tot = sum(lines(sd / f) for f in ('index.html', 'style.css', 'main.js') if (sd / f).exists())
    if tot > 450: errs.append(f'site source {tot} lines > 450')
    try:
        data = {k: [r['fid'] for v in d['front'].values() for r in v] for k, d in json.loads((sd / 'data.json').read_text())['datasets'].items()}
    except (OSError, ValueError, KeyError):
        errs.append('site/data.json missing or malformed')
    H = type('H', (SimpleHTTPRequestHandler,), {'log_message': lambda *x: None})
    srv = ThreadingHTTPServer(('127.0.0.1', 8781), functools.partial(H, directory=str(sd)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        for bn in ('firefox', 'chromium'):
            br = getattr(pw, bn).launch()
            for w, h in ((390, 844), (1440, 900)):
                e, resp = [], []
                pg = br.new_page(viewport={'width': w, 'height': h})
                pg.on('console', lambda m: m.type == 'error' and e.append('console error: ' + m.text[:80]))
                pg.on('pageerror', lambda x: e.append('page error: ' + str(x)[:80]))
                pg.on('request', lambda r: r.url.startswith(('http://127.0.0.1', 'data:', 'blob:')) or e.append('external request ' + r.url[:60]))
                pg.on('response', lambda r: resp.append(r))
                pg.goto('http://127.0.0.1:8781/index.html', wait_until='networkidle')
                pg.wait_for_timeout(400)
                e += pg.evaluate(JS, {'data': data, 'w': w})
                size = sum(len(r.body()) for r in resp if r.ok)
                if size > 1.5e6: e.append(f'transferred {size / 1e6:.2f} MB > 1.5')
                p2 = br.new_page(viewport={'width': w, 'height': h}, reduced_motion='reduce')
                p2.goto('http://127.0.0.1:8781/index.html', wait_until='networkidle')
                p2.wait_for_timeout(400)
                if p2.evaluate(ANIM): e.append('animations running under prefers-reduced-motion')
                errs += [f'[{bn} {w}x{h}] {x}' for x in e]
            br.close()
    srv.shutdown()
    return errs
def selfcheck(a):
    import numpy as np
    tmp = R / '.selfcheck_tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    for d in ('runs/stub', 'site', '.pyramid'):
        (tmp / d).mkdir(parents=True)
    env = dict(os.environ, LOOM_ROOT=str(tmp), LOOM_SELFCHECK='1')
    errs = []

    def run(*c, want=1, grep=''):
        p = subprocess.run([sys.executable, __file__, *c], env=env, capture_output=True, text=True, timeout=900)
        out = p.stdout + p.stderr
        if p.returncode != want or 'Traceback' in out or (want == 1 and not out.startswith('FAIL')) or grep not in out:
            errs.append(f'selfcheck {" ".join(c)}: exit {p.returncode} (want {want}), grep "{grep}": {out.strip()[:200]}')
        return out
    try:
        row = dict(name='stub', dataset='cifar', tag='iter1', cfg=dict(in_ch=3, img=32), params=1, active_params=1, steps=1, batch=1, lr=1., seed=0,
                   nfe=1, fid_n=300, fid=0., loss_final=0., train_min=1., it_s=1., peak_gb=1.)
        rs = tmp / 'results.jsonl'
        rs.write_text(json.dumps(dict(row, finished=datetime.now().isoformat())) + '\n')
        run('record', 'stub', grep='missing'); run('record', 'nonexistent', grep='no matching')
        x = np.random.randint(0, 256, (300, 32, 32, 3), dtype=np.uint8)
        np.savez(tmp / 'runs/stub/samples.npz', x=x); (tmp / 'runs/stub/grid.png').write_bytes(b'png')
        row.update(fid=fid(x, 'cifar'), finished=datetime.now().isoformat())
        rs.write_text(json.dumps(row) + '\n')
        run('record', 'last', want=0)
        rs.write_text(json.dumps(dict(row, fid=row['fid'] + 5)) + '\n')
        run('record', 'stub', grep='recomputed fid')
        rs.write_text(json.dumps(row) + '\n')
        (tmp / 'NOTES.md').write_text('## iter 1\nhypothesis: a\nresult: b\nnext: c\n')
        run('iter', '1', want=0); run('iter', '2', grep='no rows')
        (tmp / 'README.md').write_text('```\npython check.py unit\nPY check.py bogus\n```\n')
        run('readme', grep='bogus')
        (tmp / '.pyramid/review.md').write_text('## blockers\n## majors\n')
        run('review', grep='minors')
        [run(c) for c in ('smoke', 'pareto')]
        (tmp / 'site/index.html').write_text('<!doctype html><title>x</title><p>placeholder</p>')
        (tmp / 'site/data.json').write_text(json.dumps(dict(datasets={d: dict(front=dict(loom=[dict(name='a', fid=12.34)])) for d in ('mnist', 'cifar')})))
        out = run('site')
        errs += [f'selfcheck site: no result for {b} {v}' for b in ('firefox', 'chromium') for v in ('390x844', '1440x900') if f'[{b} {v}]' not in out]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return errs
CMDS = dict(selfcheck=selfcheck, stats=stats, unit=unit, record=record, smoke=smoke, iter=iter_, pareto=pareto, readme=readme, review=review, site=site)
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS or (sys.argv[1] == 'iter' and len(sys.argv) < 3):
        print('usage: check.py {' + '|'.join(CMDS) + '} [args]', file=sys.stderr); sys.exit(2)
    try:
        errs = CMDS[sys.argv[1]](sys.argv[2:])
    except Exception as ex:
        errs = [f'{sys.argv[1]}: {type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ""}'[:240]]
    print('\n'.join('FAIL ' + e for e in errs) or f'{sys.argv[1]}: ok')
    sys.exit(1 if errs else 0)
