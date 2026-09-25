export const isNum = v => v !== '' && v != null && !isNaN(Number(v))

export function columnInfo(rows, j) {
  const vals = rows.map(r => r[j])
  const present = vals.filter(v => v !== '' && v != null)
  return { present, missing: vals.length - present.length, numeric: present.length > 0 && present.every(isNum) }
}

export function describeCol(name, rows, j) {
  const { present, numeric, missing } = columnInfo(rows, j)
  const base = { name, type: numeric ? 'numeric' : 'categorical', count: present.length, missing, unique: new Set(present).size }
  if (numeric) {
    const x = present.map(Number).sort((a, b) => a - b), n = x.length
    const mean = x.reduce((a, b) => a + b, 0) / n
    const std = Math.sqrt(x.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(n - 1, 1))
    const q = p => { const i = (n - 1) * p, lo = Math.floor(i); return x[lo] + (x[Math.ceil(i)] - x[lo]) * (i - lo) }
    const min = x[0], max = x[n - 1], bins = Array(10).fill(0)
    x.forEach(v => { bins[Math.min(9, max === min ? 0 : Math.floor(((v - min) / (max - min)) * 10))]++ })
    return { ...base, mean, std, min, q1: q(.25), median: q(.5), q3: q(.75), max, bins }
  }
  const freq = {}
  present.forEach(v => (freq[v] = (freq[v] || 0) + 1))
  const top = Object.entries(freq).sort((a, b) => b[1] - a[1])
  return { ...base, top: top[0]?.[0], topFreq: top[0]?.[1], freq: top.slice(0, 5) }
}

export function correlation(rows, a, b) {
  const p = rows.filter(r => isNum(r[a]) && isNum(r[b])).map(r => [+r[a], +r[b]])
  const n = p.length; if (n < 2) return 0
  const ma = p.reduce((s, x) => s + x[0], 0) / n, mb = p.reduce((s, x) => s + x[1], 0) / n
  let sab = 0, sa = 0, sb = 0
  p.forEach(([x, y]) => { sab += (x - ma) * (y - mb); sa += (x - ma) ** 2; sb += (y - mb) ** 2 })
  return sa && sb ? sab / Math.sqrt(sa * sb) : 0
}

export function detectTask(rows, j) {
  const { present, numeric } = columnInfo(rows, j)
  if (!numeric) return 'classification'
  const u = new Set(present)
  return u.size <= 10 && present.every(v => Number.isInteger(+v)) ? 'classification' : 'regression'
}

const rng = s => () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 }

export function prepare(rows, headers, target, task) {
  const t = headers.indexOf(target)
  let ok = rows.filter(r => r[t] !== '' && (task === 'classification' || isNum(r[t])))
  const rand = rng(42)
  ok = ok.map(r => [rand(), r]).sort((a, b) => a[0] - b[0]).slice(0, 5000).map(x => x[1]) // shuffle + cap
  const fns = []
  headers.forEach((h, j) => {
    if (j === t) return
    const { numeric, present } = columnInfo(ok, j)
    if (!present.length) return
    if (numeric) {
      const x = present.map(Number), m = x.reduce((a, b) => a + b, 0) / x.length
      const s = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / x.length) || 1
      fns.push(r => [isNum(r[j]) ? (Number(r[j]) - m) / s : 0])
    } else {
      const cats = [...new Set(present)]
      if (cats.length <= 20) fns.push(r => cats.map(c => (r[j] === c ? 1 : 0)))
    }
  })
  const X = ok.map(r => fns.flatMap(f => f(r)))
  const y = ok.map(r => (task === 'regression' ? Number(r[t]) : r[t]))
  const cut = Math.max(1, Math.floor(ok.length * 0.8))
  return { Xtr: X.slice(0, cut), ytr: y.slice(0, cut), Xte: X.slice(cut), yte: y.slice(cut), nFeat: X[0]?.length || 0 }
}

function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]])
  for (let i = 0; i < n; i++) {
    let p = i
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r
    ;[M[i], M[p]] = [M[p], M[i]]
    for (let r = i + 1; r < n; r++) { const f = M[r][i] / M[i][i]; for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c] }
  }
  const x = Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c]; x[i] = s / M[i][i] }
  return x
}

export function runModel(kind, task, d) {
  const { Xtr, ytr, Xte, yte } = d
  if (!Xtr.length || !Xte.length || !d.nFeat) throw new Error('Not enough usable data or features to train.')
  const dist = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0)
  let pred
  if (kind === 'linear') {
    const A = Xtr.map(r => [1, ...r]), p = A[0].length
    const XtX = Array.from({ length: p }, () => Array(p).fill(0)), Xty = Array(p).fill(0)
    A.forEach((r, i) => { for (let a = 0; a < p; a++) { Xty[a] += r[a] * ytr[i]; for (let b = 0; b < p; b++) XtX[a][b] += r[a] * r[b] } })
    for (let a = 1; a < p; a++) XtX[a][a] += 1e-3
    const w = solve(XtX, Xty)
    pred = Xte.map(r => w[0] + r.reduce((s, v, i) => s + v * w[i + 1], 0))
  } else if (kind === 'knn') {
    const k = Math.min(5, Xtr.length)
    pred = Xte.map(r => {
      const nn = Xtr.map((q, i) => [dist(q, r), i]).sort((a, b) => a[0] - b[0]).slice(0, k).map(x => ytr[x[1]])
      if (task === 'regression') return nn.reduce((a, b) => a + b, 0) / k
      const c = {}; nn.forEach(v => (c[v] = (c[v] || 0) + 1))
      return Object.entries(c).sort((a, b) => b[1] - a[1])[0][0]
    })
  } else {
    const g = {}
    Xtr.forEach((r, i) => { const o = (g[ytr[i]] ||= { n: 0, s: Array(r.length).fill(0) }); o.n++; r.forEach((v, j) => (o.s[j] += v)) })
    const cs = Object.entries(g).map(([c, o]) => [c, o.s.map(v => v / o.n)])
    pred = Xte.map(r => cs.map(([c, m]) => [c, dist(m, r)]).sort((a, b) => a[1] - b[1])[0][0])
  }
  if (task === 'regression') {
    const m = yte.reduce((a, b) => a + b, 0) / yte.length
    const ssr = yte.reduce((s, v, i) => s + (v - pred[i]) ** 2, 0), sst = yte.reduce((s, v) => s + (v - m) ** 2, 0)
    return { pred, actual: yte, metrics: { 'R²': sst ? 1 - ssr / sst : 0, RMSE: Math.sqrt(ssr / yte.length), MAE: yte.reduce((s, v, i) => s + Math.abs(v - pred[i]), 0) / yte.length }, train: Xtr.length, test: Xte.length }
  }
  const classes = [...new Set([...ytr, ...yte])].sort()
  const cm = classes.map(() => classes.map(() => 0))
  yte.forEach((v, i) => cm[classes.indexOf(v)][classes.indexOf(pred[i])]++)
  return { classes, cm, metrics: { Accuracy: yte.filter((v, i) => v === pred[i]).length / yte.length }, train: Xtr.length, test: Xte.length }
}
