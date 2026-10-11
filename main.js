const NS = 'http://www.w3.org/2000/svg';
const NAMES = { mnist: 'MNIST', cifar: 'CIFAR-10' };
const M = n => (n / 1e6).toFixed(2) + 'M';
const f1 = x => x.toFixed(1);
const FAM = { dense: 'dense', loom: 'LOOM' };

function s(tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  return e;
}
function h(tag, props = {}, kids = []) {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...kids);
  return e;
}

function chart(svg, front) {
  const W = Math.max(280, Math.round(svg.parentNode.clientWidth)), H = Math.round(W * 0.66);
  if (svg.dataset.w == W) return;
  svg.dataset.w = W;
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const m = { l: 40, r: 26, t: 26, b: 40 };
  const pts = [...front.dense, ...front.loom], fids = pts.map(p => p.fid), ps = pts.map(p => p.params);
  const lo = Math.log10(Math.min(...ps) / 1.4), hi = Math.log10(Math.max(...ps) * 1.4);
  const step = Math.max(...fids) - Math.min(...fids) > 60 ? 20 : 10;
  const y0 = Math.floor((Math.min(...fids) - step / 2) / step) * step;
  const y1 = Math.ceil((Math.max(...fids) + step / 2) / step) * step;
  const X = p => m.l + (Math.log10(p) - lo) / (hi - lo) * (W - m.l - m.r);
  const Y = f => H - m.b - (f - y0) / (y1 - y0) * (H - m.t - m.b);
  for (let f = y0; f <= y1; f += step)
    svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: Y(f), y2: Y(f) }),
      s('text', { x: m.l - 8, y: Y(f) + 4, 'text-anchor': 'end' }, f));
  for (const t of [1e5, 3e5, 1e6, 3e6, 1e7, 3e7]) {
    if (Math.log10(t) < lo || Math.log10(t) > hi) continue;
    svg.append(s('line', { class: 'ax', x1: X(t), x2: X(t), y1: H - m.b, y2: H - m.b + 5 }),
      s('text', { x: X(t), y: H - m.b + 18, 'text-anchor': 'middle' }, t >= 1e6 ? t / 1e6 + 'M' : t / 1e3 + 'k'));
  }
  svg.append(s('text', { x: W - m.r, y: H - 4, 'text-anchor': 'end' }, 'parameters, log scale'),
    s('text', { x: m.l - 8, y: m.t - 12, 'text-anchor': 'end' }, 'FID'));
  for (const fam of ['dense', 'loom']) {
    svg.append(s('polyline', { class: fam, points: front[fam].map(p => `${X(p.params)},${Y(p.fid)}`).join(' ') }));
    for (const p of front[fam]) {
      const c = s('circle', { class: fam, cx: X(p.params), cy: Y(p.fid), r: 3.5 });
      c.append(s('title', {}, `${FAM[fam]}, ${M(p.params)} parameters, FID ${f1(p.fid)}`));
      svg.append(c, s('text', { class: fam, x: X(p.params), y: Y(p.fid) + (fam === 'loom' ? -9 : 17), 'text-anchor': 'middle' }, f1(p.fid)));
    }
  }
}

function table(front) {
  const rows = ['dense', 'loom'].flatMap(f => front[f].map(p => [p, f])).sort((a, b) => a[0].params - b[0].params);
  const cell = (p, f, want) => h('td', { className: want, textContent: f === want ? f1(p.fid) : '' });
  return h('table', {}, [
    h('thead', {}, [h('tr', {}, ['parameters', 'dense FID', 'LOOM FID'].map(t => h('th', { textContent: t })))]),
    h('tbody', {}, rows.map(([p, f]) => h('tr', {}, [h('td', { textContent: M(p.params) }), cell(p, f, 'dense'), cell(p, f, 'loom')]))),
  ]);
}

function gallery(ds, front) {
  return [0, -1].flatMap(i => ['dense', 'loom'].map(f => {
    const p = front[f].at(i);
    return h('figure', {}, [
      h('img', { src: p.grid, width: 1024, height: 1024, decoding: 'async', alt: `${NAMES[ds]} samples, ${FAM[f]}, ${M(p.params)} parameters` }),
      h('figcaption', {}, [h('span', { className: f, textContent: FAM[f] }), ` ${M(p.params)} · FID ${f1(p.fid)}`]),
    ]);
  }));
}

async function main() {
  const d = await (await fetch('data.json')).json();
  const draws = [];
  for (const [ds, v] of Object.entries(d.datasets)) {
    const fig = document.getElementById(`fig-${ds}`);
    const svg = s('svg', { 'data-chart': ds, role: 'img', 'aria-label': `${NAMES[ds]}: FID against parameters, dense and LOOM fronts` });
    const run = v.all.find(r => r.name === v.front.dense[0].name);
    if (run) fig.querySelector('.steps').textContent = `${run.steps.toLocaleString('en')} steps`;
    fig.append(svg, table(v.front));
    document.getElementById(`gal-${ds}`).append(...gallery(ds, v.front));
    draws.push(() => chart(svg, v.front));
  }
  document.getElementById('gen').textContent = d.generated.slice(0, 10);
  const draw = () => draws.forEach(f => f());
  draw();
  let t;
  addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 150); });
}
main();
