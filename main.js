const NS = 'http://www.w3.org/2000/svg';
const NAMES = { mnist: 'MNIST', cifar: 'CIFAR-10' };
const FAM = { looped: 'looped dense', loom: 'LOOM', dense: 'dense' };
const M = n => (n / 1e6).toFixed(2) + 'M';
const f1 = x => x.toFixed(1);
const fams = front => Object.keys(FAM).filter(f => front[f]);
const SPOTS = { looped: [[0, 17, 'middle'], [0, -9, 'middle']], loom: [[0, -9, 'middle'], [0, 17, 'middle']], dense: [] };
const SIDES = [[8, 4, 'start'], [-8, 4, 'end']];

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
const hit = (a, b) => a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];

function chart(svg, front) {
  const W = Math.max(280, Math.round(svg.parentNode.clientWidth)), H = Math.round(W * 0.7);
  if (svg.dataset.w == W) return;
  svg.dataset.w = W;
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const m = { l: 40, r: 30, t: 26, b: 40 };
  const pts = fams(front).flatMap(f => front[f].map(p => [p, f])), fids = pts.map(([p]) => p.fid), ps = pts.map(([p]) => p.params);
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
  for (const f of fams(front))
    svg.append(s('polyline', { class: f, points: front[f].map(p => `${X(p.params)},${Y(p.fid)}`).join(' ') }));
  const boxes = pts.map(([p]) => [X(p.params) - 5, Y(p.fid) - 5, X(p.params) + 5, Y(p.fid) + 5]);
  for (const [p, f] of pts) {
    const c = s('circle', { class: f, cx: X(p.params), cy: Y(p.fid), r: 3.5 });
    c.append(s('title', {}, `${FAM[f]}, ${M(p.params)} parameters, FID ${f1(p.fid)}`));
    svg.append(c);
  }
  for (const [p, f] of pts) {
    const x = X(p.params), y = Y(p.fid), w = f1(p.fid).length * 6.7;
    const box = ([dx, dy, a]) => { const x0 = x + dx - (a === 'middle' ? w / 2 : a === 'end' ? w : 0); return [x0, y + dy - 9, x0 + w, y + dy + 2]; };
    const opts = [...SPOTS[f], ...SIDES, ...SPOTS.looped];
    const [dx, dy, a] = opts.find(o => !boxes.some(b => hit(b, box(o)))) || opts[0];
    boxes.push(box([dx, dy, a]));
    svg.append(s('text', { class: `lab ${f}`, x: x + dx, y: y + dy, 'text-anchor': a }, f1(p.fid)));
  }
}

function table(front) {
  const rows = new Map();
  for (const f of fams(front)) for (const p of front[f]) {
    const k = M(p.params);
    rows.set(k, { ...rows.get(k), [f]: p, n: p.params });
  }
  return h('table', {}, [
    h('thead', {}, [h('tr', {}, ['parameters', ...fams(front).map(f => FAM[f] + ' FID')].map(t => h('th', { textContent: t })))]),
    h('tbody', {}, [...rows].sort((a, b) => a[1].n - b[1].n).map(([k, r]) => h('tr', {}, [
      h('td', { textContent: k }), ...fams(front).map(f => h('td', { className: f, textContent: r[f] ? f1(r[f].fid) : '' }))]))),
  ]);
}

function gallery(ds, front) {
  return fams(front).map(f => {
    const p = front[f].reduce((a, b) => Math.abs(Math.log(b.params / 9e6)) < Math.abs(Math.log(a.params / 9e6)) ? b : a);
    return h('figure', {}, [
      h('img', { src: p.grid, width: 1024, height: 1024, decoding: 'async', alt: `${NAMES[ds]} samples, ${FAM[f]}, ${M(p.params)} parameters` }),
      h('figcaption', {}, [h('span', { className: f, textContent: FAM[f] }), ` ${M(p.params)} · FID ${f1(p.fid)}`]),
    ]);
  });
}

async function main() {
  const d = await (await fetch('data.json')).json();
  const draws = [];
  for (const [ds, v] of Object.entries(d.datasets)) {
    const fig = document.getElementById(`fig-${ds}`);
    const svg = s('svg', { 'data-chart': ds, role: 'img', 'aria-label': `${NAMES[ds]}: FID against parameters, ${fams(v.front).map(f => FAM[f]).join(', ')} fronts` });
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
