import { useEffect, useMemo, useState } from 'react'
import Papa from 'papaparse'
import { describeCol, correlation, detectTask, prepare, runModel } from './ml'

const fmt = v => (typeof v !== 'number' ? v ?? '—' : Number.isNaN(v) ? '—' : Number.isInteger(v) ? v : +v.toFixed(3))
const PAL = ['#0b7a75', '#e4572e', '#3b5bdb', '#c47f00', '#7b2cbf', '#6b7280']
const NAV = [['overview', 'Overview'], ['data', 'Data'], ['stats', 'Statistics'], ['charts', 'Charts'], ['model', 'Model']]
const MODELS = {
  regression: [['linear', 'Linear regression', 'Fast, easy-to-read baseline for straight-line relationships.'], ['knn', 'K-nearest neighbors', 'Averages the 5 most similar rows.']],
  classification: [['knn', 'K-nearest neighbors', 'Majority vote of the 5 most similar rows.'], ['centroid', 'Nearest centroid', 'Picks the class whose average row is closest.']],
}


export default function App() {
  const [raw, setRaw] = useState(null)
  const [header, setHeader] = useState(true)
  const [tab, setTab] = useState('overview')
  const [target, setTarget] = useState('')
  const [task, setTask] = useState('')
  const [result, setResult] = useState(null)
  const [limit, setLimit] = useState(10)
  const [q, setQ] = useState('')
  const [sort, setSort] = useState(null)
  const [drag, setDrag] = useState(false)
  const [err, setErr] = useState('')
  const [dark, setDark] = useState(() => matchMedia('(prefers-color-scheme: dark)').matches)
  const [xc, setXc] = useState(-1), [yc, setYc] = useState(-1), [cc, setCc] = useState(-1)
  useEffect(() => { document.documentElement.dataset.theme = dark ? 'dark' : 'light' }, [dark])

  const { headers, rows } = useMemo(() => {
    if (!raw) return { headers: [], rows: [] }
    const w = Math.max(...raw.rows.map(r => r.length))
    const all = raw.rows.map(r => Array.from({ length: w }, (_, i) => String(r[i] ?? '').trim()))
    if (header) return { headers: all[0].map((x, i) => x || `column_${i + 1}`), rows: all.slice(1) }
    return { headers: all[0].map((_, i) => `column_${i + 1}`), rows: all }
  }, [raw, header])

  const stats = useMemo(() => headers.map((h, j) => describeCol(h, rows, j)), [headers, rows])
  const numIdx = stats.map((s, i) => (s.type === 'numeric' ? i : -1)).filter(i => i >= 0)
  const catIdx = stats.map((s, i) => (s.type !== 'numeric' && s.unique <= 6 ? i : -1)).filter(i => i >= 0)
  const dups = useMemo(() => rows.length - new Set(rows.map(r => r.join('\u0001'))).size, [rows])
  const corr = useMemo(() => numIdx.map(a => numIdx.map(b => (a === b ? 1 : correlation(rows, a, b)))), [rows, numIdx.join()])
  const missing = stats.reduce((s, c) => s + c.missing, 0)

  const insights = useMemo(() => {
    const o = []
    stats.forEach(s => {
      if (s.missing) o.push(['warn', `${s.name} is missing ${s.missing} values (${((s.missing / rows.length) * 100).toFixed(1)}%). Training fills them in automatically.`])
      if (s.unique === 1) o.push(['warn', `${s.name} has the same value in every row, so a model can't learn from it.`])
      if (s.type !== 'numeric' && s.unique > 20) o.push(['warn', `${s.name} has ${s.unique} distinct values, so it is left out of training.`])
    })
    if (dups) o.push(['warn', `${dups} duplicate rows found.`])
    numIdx.forEach((a, x) => numIdx.forEach((b, y) => { if (y > x && Math.abs(corr[x]?.[y]) > .8) o.push(['info', `${headers[a]} and ${headers[b]} move closely together (r = ${corr[x][y].toFixed(2)}).`]) }))
    if (!o.length) o.push(['ok', 'No problems found. The data looks ready for modeling.'])
    return o
  }, [stats, dups, corr])

  const tgt = headers.includes(target) ? target : headers[headers.length - 1]
  const tk = task || (tgt ? detectTask(rows, headers.indexOf(tgt)) : 'regression')
  const X = numIdx.includes(xc) ? xc : numIdx[0], Y = numIdx.includes(yc) ? yc : numIdx[1] ?? numIdx[0]

  const start = (name, size, data) => { setRaw({ name, size, rows: data }); setTarget(''); setTask(''); setResult(null); setTab('overview'); setSort(null); setQ('') }
  const load = file => {
    setErr('')
    Papa.parse(file, {
      skipEmptyLines: 'greedy',
      complete: r => (r.data.length > 1 ? start(file.name, file.size, r.data) : setErr('This file has no data rows. Check that it is a CSV with a header line.')),
      error: e => setErr(e.message),
    })
  }
  const train = kind => {
    setErr('')
    try { setResult({ kind, ...runModel(kind, tk, prepare(rows, headers, tgt, tk)) }) } catch (e) { setResult(null); setErr(e.message) }
  }
  const download = () => {
    const csv = Papa.unparse(stats.map(({ bins, freq, ...s }) => s))
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); a.download = 'statistics.csv'; a.click()
  }

  const shown = useMemo(() => {
    let d = rows.map((r, i) => [i + 1, r]).filter(([, r]) => !q || r.some(c => c.toLowerCase().includes(q.toLowerCase())))
    if (sort) { const num = stats[sort.j]?.type === 'numeric'; d = [...d].sort(([, a], [, b]) => (num ? a[sort.j] - b[sort.j] : a[sort.j].localeCompare(b[sort.j])) * sort.dir) }
    return d
  }, [rows, q, sort, stats])

  const dropzone = (
    <label className={'drop' + (drag ? ' on' : '')} onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); e.dataTransfer.files[0] && load(e.dataTransfer.files[0]) }}>
      <input type="file" accept=".csv,.tsv,.txt" hidden onChange={e => e.target.files[0] && load(e.target.files[0])} />
      <b>{raw ? 'Replace file' : 'Drop a CSV file here'}</b><span>or click to browse your computer</span>
    </label>
  )

  return (
    <div className="shell">
        <header className="topbar">
    <div className="top">
      <div className="brand"><i />CSV ML Explorer</div>
      <button className="theme" onClick={() => setDark(!dark)}
        aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        title={dark ? 'Switch to light mode' : 'Switch to dark mode'}>{dark ? '☀' : '☾'}</button>
    </div>
    {raw && <nav>{NAV.map(([k, n]) => <button key={k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{n}</button>)}</nav>}
    {raw && <div className="file"><span title={raw.name}><b>{raw.name}</b> · {rows.length} rows · {headers.length} columns</span>{dropzone}</div>}
  </header>

      <main>
        {err && <div className="err" role="alert">{err}</div>}

        {!raw && <div className="empty">
          <h1>See what's in your data, then train a model on it.</h1>
          <p>Upload a CSV to get a preview, statistics, charts and a quick machine-learning baseline.</p>
          {dropzone}
        
        </div>}

        {raw && tab === 'overview' && <div className="page">
          <h1>{raw.name}</h1>
          <dl className="kpis">{[['Rows', rows.length], ['Columns', headers.length], ['Numeric', numIdx.length], ['Categorical', headers.length - numIdx.length], ['Missing cells', missing], ['Duplicates', dups]]
            .map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
          <div className="two">
            <section><h2>Column health</h2>
              {stats.map(s => <div className="health" key={s.name}><span title={s.name}>{s.name}</span><em>{s.type}</em>
                <div className="meter" title={`${s.count} of ${rows.length} filled`}><i style={{ width: `${(s.count / rows.length) * 100}%` }} /></div><small>{s.unique} unique</small></div>)}
            </section>
            <section><h2>Worth a look</h2>
              <ul className="notes">{insights.map(([k, t], i) => <li key={i} className={k}>{t}</li>)}</ul></section>
          </div>
        </div>}

        {raw && tab === 'data' && <div className="page">
          <h1>Data</h1>
          <div className="bar">
            <input placeholder="Search all cells" aria-label="Search rows" value={q} onChange={e => setQ(e.target.value)} />
            <select aria-label="Rows to show" value={limit} onChange={e => setLimit(+e.target.value)}>{[10, 25, 50, 100].map(n => <option key={n} value={n}>Show {n}</option>)}</select>
            <div className="seg"><span>First row is header</span>
              <button className={header ? 'on' : ''} onClick={() => setHeader(true)}>Yes</button><button className={!header ? 'on' : ''} onClick={() => setHeader(false)}>No</button></div>
          </div>
          <section className="flush"><div className="scroll"><table><thead><tr><th>#</th>{headers.map((h, j) =>
            <th key={j}><button className="th" onClick={() => setSort(sort?.j === j ? (sort.dir === 1 ? { j, dir: -1 } : null) : { j, dir: 1 })}>{h}{sort?.j === j ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}</button></th>)}</tr></thead>
            <tbody>{shown.slice(0, limit).map(([i, r]) => <tr key={i}><td className="idx">{i}</td>{r.map((c, j) => <td key={j} className={c === '' ? 'na' : stats[j]?.type === 'numeric' ? 'num' : ''}>{c === '' ? 'missing' : c}</td>)}</tr>)}</tbody></table></div></section>
          <small>Showing {Math.min(limit, shown.length)} of {shown.length} matching rows.</small>
        </div>}

        {raw && tab === 'stats' && <div className="page">
          <div className="bar"><h1>Statistics</h1><button onClick={download}>Download as CSV</button></div>
          <section className="flush"><div className="scroll"><table><thead><tr>{['Column', 'Type', 'Count', 'Missing', 'Unique', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Most common'].map(h => <th key={h}>{h}</th>)}</tr></thead>
            <tbody>{stats.map(s => <tr key={s.name}><td><b>{s.name}</b></td><td>{s.type}</td><td className="num">{s.count}</td><td className={'num' + (s.missing ? ' na' : '')}>{s.missing}</td><td className="num">{s.unique}</td>
              {['mean', 'std', 'min', 'q1', 'median', 'q3', 'max'].map(k => <td className="num" key={k}>{fmt(s[k])}</td>)}<td>{s.top ? `${s.top} (${s.topFreq})` : '—'}</td></tr>)}</tbody></table></div></section>
        </div>}

        {raw && tab === 'charts' && <div className="page">
          <h1>Charts</h1>
          {numIdx.length > 0 && <section><div className="bar"><h2>Compare two columns</h2>
            <label>X<select value={X} onChange={e => setXc(+e.target.value)}>{numIdx.map(i => <option key={i} value={i}>{headers[i]}</option>)}</select></label>
            <label>Y<select value={Y} onChange={e => setYc(+e.target.value)}>{numIdx.map(i => <option key={i} value={i}>{headers[i]}</option>)}</select></label>
            <label>Color by<select value={cc} onChange={e => setCc(+e.target.value)}><option value={-1}>None</option>{catIdx.map(i => <option key={i} value={i}>{headers[i]}</option>)}</select></label></div>
            <Scatter rows={rows} X={X} Y={Y} cc={cc} xn={headers[X]} yn={headers[Y]} /></section>}
          <section><h2>Distributions</h2><div className="dists">{stats.map(s => <div className="dist" key={s.name}><b>{s.name}</b>
            {s.bins ? <><div className="bars">{s.bins.map((n, i) => <i key={i} style={{ height: `${(n / Math.max(...s.bins)) * 100}%` }} title={`${n} rows`} />)}</div><small>{fmt(s.min)} to {fmt(s.max)}</small></>
              : s.freq.map(([k, n]) => <div className="hb" key={k}><span title={k}>{k}</span><i style={{ width: `${(n / s.count) * 100}%` }} /><em>{n}</em></div>)}</div>)}</div></section>
          {numIdx.length > 1 && <section><h2>Correlation</h2><div className="scroll"><table className="heat"><thead><tr><th />{numIdx.map(i => <th key={i}>{headers[i]}</th>)}</tr></thead>
            <tbody>{numIdx.map((a, x) => <tr key={a}><th>{headers[a]}</th>{corr[x]?.map((v, y) => <td key={y} style={{ background: `rgba(${v > 0 ? '11,122,117' : '228,87,46'},${Math.abs(v) * .9})`, color: Math.abs(v) > .5 ? '#fff' : 'inherit' }}>{v.toFixed(2)}</td>)}</tr>)}</tbody></table></div></section>}
        </div>}

        {raw && tab === 'model' && <div className="page">
          <h1>Model</h1>
          <div className="two">
            <section><h2>What should it predict?</h2>
              <label>Target column<select value={tgt} onChange={e => { setTarget(e.target.value); setTask(''); setResult(null) }}>{headers.map(h => <option key={h}>{h}</option>)}</select></label>
              <div className="tasks">{[['regression', 'Regression', 'Predict a number'], ['classification', 'Classification', 'Predict a category']].map(([k, t, d]) =>
                <button key={k} className={tk === k ? 'on' : ''} onClick={() => { setTask(k); setResult(null) }}><b>{t}</b><small>{d}</small></button>)}</div>
              <small>We suggest {detectTask(rows, headers.indexOf(tgt))} for this column. You can change it. Every other column is used as input.</small></section>
            <section><h2>Pick a model</h2><div className="models">{MODELS[tk].map(([k, n, d]) =>
              <button key={k} className={result?.kind === k ? 'on' : ''} onClick={() => train(k)}><b>{n}</b><small>{d}</small><span>Train</span></button>)}</div>
              <small>80% of rows train the model and 20% test it.</small></section>
          </div>
          {result ? <Results r={result} task={tk} /> : <section className="hint">Train a model to see how well it predicts {tgt}.</section>}
        </div>}
      </main>
    </div>
  )
}

function Scatter({ rows, X, Y, cc, xn, yn }) {
  const pts = rows.filter(r => r[X] !== '' && r[Y] !== '').map(r => [+r[X], +r[Y], cc >= 0 ? r[cc] : ''])
  if (!pts.length) return null
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const cats = [...new Set(pts.map(p => p[2]))], sx = v => 60 + ((v - x0) / (x1 - x0 || 1)) * 520, sy = v => 320 - ((v - y0) / (y1 - y0 || 1)) * 290
  return <div><svg viewBox="0 0 600 360" className="plot" role="img" aria-label={`${yn} versus ${xn}`}>
    <line x1="60" y1="320" x2="580" y2="320" className="axis" /><line x1="60" y1="30" x2="60" y2="320" className="axis" />
    {pts.map((p, i) => <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r="3.5" fill={PAL[cats.indexOf(p[2]) % 6]} opacity=".65" />)}
    <text x="60" y="340" className="tick">{fmt(x0)}</text><text x="580" y="340" className="tick" textAnchor="end">{fmt(x1)}</text><text x="320" y="356" className="tick" textAnchor="middle">{xn}</text>
    <text x="54" y="320" className="tick" textAnchor="end">{fmt(y0)}</text><text x="54" y="36" className="tick" textAnchor="end">{fmt(y1)}</text><text x="10" y="14" className="tick">{yn}</text></svg>
    {cc >= 0 && <div className="legend">{cats.map((c, i) => <span key={c}><i style={{ background: PAL[i % 6] }} />{c}</span>)}</div>}</div>
}

function Results({ r, task }) {
  let plot
  if (task === 'regression') {
    const all = [...r.actual, ...r.pred], lo = Math.min(...all), hi = Math.max(...all), sc = v => 50 + ((v - lo) / (hi - lo || 1)) * 290
    plot = <svg viewBox="0 0 360 380" className="plot sq" role="img" aria-label="Predicted versus actual"><line x1="50" y1="340" x2="340" y2="50" className="axis" strokeDasharray="4" />
      {r.actual.map((a, i) => <circle key={i} cx={sc(a)} cy={390 - sc(r.pred[i])} r="3.5" fill={PAL[0]} opacity=".7" />)}
      <text x="195" y="372" className="tick" textAnchor="middle">Actual</text><text x="12" y="195" className="tick" transform="rotate(-90 12 195)" textAnchor="middle">Predicted</text></svg>
  } else {
    plot = <div className="scroll"><table className="heat"><thead><tr><th>Actual ↓  Predicted →</th>{r.classes.map(c => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{r.cm.map((row, i) => <tr key={i}><th>{r.classes[i]}</th>{row.map((v, j) => <td key={j} style={{ background: i === j ? 'rgba(11,122,117,.35)' : v ? 'rgba(228,87,46,.35)' : '' }}>{v}</td>)}</tr>)}</tbody></table></div>
  }
  const note = task === 'regression' ? `R² is how much of the variation the model explains: 1 is perfect, 0 is no better than guessing the average.` : `Accuracy is the share of test rows predicted correctly. Green cells are correct, orange cells are mistakes.`
  return <section><h2>Results on {r.test} test rows</h2>
    <dl className="kpis">{Object.entries(r.metrics).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{fmt(v)}</dd></div>)}<div><dt>Trained on</dt><dd>{r.train} rows</dd></div></dl>
    {plot}<small>{note}</small></section>
}
