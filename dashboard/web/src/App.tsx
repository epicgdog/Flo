import { marked } from 'marked'
import { useMemo, useState, type ReactNode } from 'react'
import {
  C, Plot, Table, fmt, parseTs, skelTrace, statusColor, usePoll,
  type Outcome, type Status, type Summary,
} from './lib'

const TABS = ['Progress', 'Live run', 'Attempts', 'Skills', '3D viewer', 'Snap and cost', 'Drone sim'] as const
type Tab = (typeof TABS)[number]
const KINDS: Status[] = ['accepted', 'rejected', 'killed_early', 'failed']

export default function App() {
  const { data, error } = usePoll<Summary>('/api/summary', 5000)
  const [tab, setTab] = useState<Tab>(() => { const q = new URLSearchParams(location.search).get('tab'); return (TABS.find(t => t === q) ?? (localGet('tab') as Tab)) || 'Progress' })
  const pick = (t: Tab) => { setTab(t); localSet('tab', t) }
  return (
    <div className="app">
      <Header s={data} error={error} />
      <nav className="tabs">
        {TABS.map(t => <button key={t} className={t === tab ? 'active' : ''} onClick={() => pick(t)}>{t}</button>)}
      </nav>
      <div className="tab">
        {!data ? <div className="info">{error ? `Cannot reach API: ${error}` : 'Loading…'}</div> : (
          <>
            {tab === 'Progress' && <ProgressTab s={data} />}
            {tab === 'Live run' && <LiveTab s={data} />}
            {tab === 'Attempts' && <AttemptsTab s={data} />}
            {tab === 'Skills' && <SkillsTab s={data} />}
            {tab === '3D viewer' && <ViewerTab s={data} />}
            {tab === 'Snap and cost' && <SnapCostTab s={data} />}
            {tab === 'Drone sim' && <SimTab s={data} />}
          </>
        )}
      </div>
    </div>
  )
}

function localGet(k: string) { try { return localStorage.getItem('flo.' + k) } catch { return null } }
function localSet(k: string, v: string) { try { localStorage.setItem('flo.' + k, v) } catch { /* private mode */ } }

// ---------- header ----------
function Metric({ label, value, delta, tone }: { label: string; value: ReactNode; delta?: ReactNode; tone?: 'good' | 'bad' }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {delta != null && <div className={`delta ${tone ?? ''}`}>{delta}</div>}
    </div>
  )
}

function vsBase(a?: number | null, b?: number | null) {
  if (a == null || !b) return { text: 'n/a' }
  return { text: `${a - b >= 0 ? '+' : ''}${(a - b).toFixed(1)} mm (${(100 * (a - b) / b).toFixed(1)}%)`, tone: (a < b ? 'good' : 'bad') as 'good' | 'bad' }
}

function Header({ s, error }: { s: Summary | null; error: string | null }) {
  const st = s?.state ?? {}
  const now = s ? new Date(s.server_time) : new Date()
  const last = parseTs(st.last_update), ends = parseTs(st.ends_at), started = parseTs(st.started_at)
  let status = 'no run yet', color = C.grey
  if (error && !s) { status = 'API down'; color = C.red }
  else if (ends && now > ends) status = 'finished'
  else if (last && (now.getTime() - last.getTime()) / 1000 < 60) { status = 'running'; color = C.green }
  else if (st.run_id) { status = 'stalled'; color = C.red }
  const el = started ? (now.getTime() - started.getTime()) / 60000 : 0
  const rem = ends ? Math.max(0, (ends.getTime() - now.getTime()) / 60000) : 0
  const best = st.best ?? {}, cnt = st.counts ?? {}
  const d = vsBase(best.dev, s?.baseline.dev), h = vsBase(best.heldout, s?.baseline.heldout)
  return (
    <>
      <h1>Flo <span className="status" style={{ color }}>● {status}</span>
        {error && s && <span className="small error">refresh failed: {error}</span>}</h1>
      <div className="metrics">
        <Metric label="Time" value={`${el.toFixed(0)} min elapsed`} delta={`${rem.toFixed(0)} min remaining`} />
        <Metric label="Now" value={`${st.iter_id ?? '–'} · ${st.phase ?? '–'}`} delta={st.strategy ?? ''} />
        <Metric label={`Best dev${best.tag ? ` (${best.tag})` : ''}`} value={best.dev != null ? `${fmt(best.dev)} mm` : '–'} delta={d.text} tone={d.tone} />
        <Metric label="Best held-out" value={best.heldout != null ? `${fmt(best.heldout)} mm` : '–'} delta={h.text} tone={h.tone} />
        <Metric label="Iterations" value={cnt.iters ?? 0} delta={`${cnt.accepted ?? 0} accepted / ${cnt.rejected ?? 0} rejected`} />
        <Metric label="Killed early / errors" value={`${cnt.killed_early ?? 0} / ${cnt.errors ?? 0}`} />
      </div>
    </>
  )
}

// ---------- progress ----------
function ProgressTab({ s }: { s: Summary }) {
  const outs = s.attempts
  const [logy, setLogy] = useState(true)
  const strategies = useMemo(() => [...new Set(outs.map(o => o.strategy).filter(Boolean) as string[])].sort(), [outs])
  const [off, setOff] = useState<Set<string>>(new Set())
  if (!outs.length) return <div className="info">No attempts yet.</div>
  const { dev: bdev, heldout: bho } = s.baseline
  const devs = outs.map(o => o.dev).filter((v): v is number => v != null).concat(bdev ? [bdev] : [])
  const top = devs.length ? Math.max(...devs) * 1.05 : 1

  const scatter = KINDS.map(k => {
    const pts = outs.filter(o => o.status === k)
    return pts.length && {
      type: 'scatter', mode: 'markers', name: k,
      x: pts.map(o => +o.iter_id), y: pts.map(o => o.dev ?? top),
      marker: { color: statusColor(k), symbol: k === 'failed' ? 'x' : 'circle', size: 10 },
      text: pts.map(o => `${o.iter_id} ${o.strategy}: ${o.reason ?? o.tag ?? ''}`),
      hovertemplate: '%{text}<br>dev %{y:.2f} mm<extra></extra>',
    }
  }).filter(Boolean) as object[]
  let bd = s.base_outcome?.dev ?? null, bh = s.base_outcome?.heldout ?? null
  const xs: number[] = [], yd: (number | null)[] = [], yh: (number | null)[] = []
  for (const o of outs) {
    if (o.accepted) { bd = o.dev ?? bd; bh = o.heldout ?? bh }
    xs.push(+o.iter_id); yd.push(bd); yh.push(bh)
  }
  const shapes: object[] = [], ann: object[] = []
  for (const [v, c, t] of [[bdev, C.green, 'baseline dev'], [bho, C.blue, 'baseline held-out']] as const) {
    if (!v) continue
    shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: v, y1: v, line: { dash: 'dash', color: c, width: 1 } })
    ann.push({ xref: 'paper', x: 1, y: v, text: t, showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { color: c, size: 11 } })
  }
  const progress = [...scatter,
    { type: 'scatter', mode: 'lines', x: xs, y: yd, name: 'best-so-far dev', line: { shape: 'hv', color: C.green } },
    { type: 'scatter', mode: 'lines', x: xs, y: yh, name: 'best-so-far held-out', line: { shape: 'hv', color: C.blue } }]

  const accepted = outs.filter(o => o.accepted).map(o => ({
    tag: o.tag, iteration: o.iter_id, rationale: o.rationale,
    'dev change (mm)': o.best_dev_before == null || o.dev == null ? null : +(o.dev - o.best_dev_before).toFixed(2),
    'held-out change (mm)': o.best_heldout_before == null || o.heldout == null ? null : +(o.heldout - o.best_heldout_before).toFixed(2),
  }))

  const bestIter = s.state.best?.iter_id
  const liveIter = String(s.state.phase ?? '').startsWith('training') ? s.state.iter_id : undefined
  const lossA: object[] = []
  const v0 = s.loss_logs['0000']
  if (v0?.length) lossA.push({ type: 'scatter', mode: 'lines', x: v0.map(r => r[0]), y: v0.map(r => r[1]), name: 'v0', line: { dash: 'dash', color: C.ink } })
  for (const o of outs) {
    if (o.strategy && off.has(o.strategy)) continue
    const lg = s.loss_logs[o.iter_id]
    if (!lg?.length) continue
    const isBest = o.iter_id === bestIter, isLive = o.iter_id === liveIter
    lossA.push({
      type: 'scatter', mode: 'lines', name: o.iter_id, x: lg.map(r => r[0]), y: lg.map(r => r[1]),
      line: { width: isBest ? 4 : 1.5, color: isLive ? C.orange : o.accepted ? C.green : C.red },
      opacity: isBest || isLive || o.accepted ? 1 : 0.25,
    })
  }
  if (liveIter && !outs.some(o => o.iter_id === liveIter) && s.loss_logs[liveIter]?.length) {
    const lg = s.loss_logs[liveIter]
    lossA.push({ type: 'scatter', mode: 'lines', name: `${liveIter} live`, x: lg.map(r => r[0]), y: lg.map(r => r[1]), line: { width: 3, color: C.orange } })
  }

  const started = parseTs(s.state.started_at)
  const lossB: object[] = []
  for (const k of KINDS) {
    const x: number[] = [], y: number[] = [], dv: (number | null)[] = [], sym: string[] = [], txt: string[] = []
    for (const o of outs) {
      const lg = s.loss_logs[o.iter_id], tEnd = parseTs(s.iter_end[o.iter_id])
      if (o.status !== k || !lg?.length || !tEnd || !started) continue
      x.push((tEnd.getTime() - started.getTime()) / 60000)
      y.push(lg[lg.length - 1][1]); dv.push(o.dev ?? null)
      sym.push(o.strategy === 'loss_and_target' ? 'circle-open' : 'circle'); txt.push(o.iter_id)
    }
    if (!x.length) continue
    lossB.push({ type: 'scatter', mode: 'markers', name: `final loss (${k})`, x, y, text: txt, marker: { color: statusColor(k), symbol: sym, size: 10 } })
    lossB.push({ type: 'scatter', mode: 'markers', name: `dev (${k})`, x, y: dv, text: txt, yaxis: 'y2', marker: { color: statusColor(k), symbol: 'diamond', size: 7 }, opacity: 0.6 })
  }

  return (
    <>
      <div className="card">
        <Plot height={380} rev="progress" data={progress}
          layout={{ xaxis: { title: { text: 'iteration' } }, yaxis: { title: { text: 'score (mm)' } }, shapes, annotations: ann, margin: { t: 10, r: 20, b: 85, l: 55 } }} />
      </div>
      <h2>Accepted changes</h2>
      <Table rows={accepted} />
      <h2>Loss over time</h2>
      <div className="controls">
        <label><input type="checkbox" checked={logy} onChange={e => setLogy(e.target.checked)} /> log y-axis</label>
        <div className="chips">
          {strategies.map(st => (
            <button key={st} className={'chip ' + (off.has(st) ? '' : 'on')}
              onClick={() => setOff(p => { const n = new Set(p); if (n.has(st)) n.delete(st); else n.add(st); return n })}>{st}</button>
          ))}
        </div>
      </div>
      <div className="card">
        <Plot rev="lossA" data={lossA} layout={{
          title: { text: 'A: training loss vs elapsed seconds', font: { size: 13 } },
          xaxis: { title: { text: 'elapsed s' } }, yaxis: { title: { text: 'loss' }, type: logy ? 'log' : 'linear' },
          margin: { t: 40, r: 20, b: 85, l: 55 }, showlegend: false,
        }} />
      </div>
      <div className="card">
        <Plot rev="lossB" data={lossB} layout={{
          title: { text: 'B: final loss across the run (hollow = loss_and_target, different loss scale)', font: { size: 13 } },
          xaxis: { title: { text: 'minutes since loop start' } }, yaxis: { title: { text: 'final training loss' } },
          yaxis2: { title: { text: 'dev score (mm)' }, overlaying: 'y', side: 'right' }, margin: { t: 40, r: 60, b: 85, l: 55 },
        }} />
      </div>
    </>
  )
}

// ---------- live ----------
function LiveTab({ s }: { s: Summary }) {
  const st = s.state, lg = s.live.log
  const el = st.train_elapsed_s ?? 0
  return (
    <>
      <p><b>{st.iter_id ?? '–'}</b> · {st.phase ?? '–'} · {st.strategy ?? ''} — <span className="muted">{st.rationale ?? ''}</span></p>
      <div className="progress"><div style={{ width: `${Math.min(100, (100 * el) / s.budget_s)}%` }} /></div>
      <div className="small muted">training {el.toFixed(0)} / {s.budget_s} s, loss {st.train_loss != null ? fmt(st.train_loss, 4) : '–'}</div>
      {lg.length > 0 && (
        <div className="card" style={{ marginTop: 12 }}>
          <Plot height={280} rev="live" data={[{ type: 'scatter', mode: 'lines+markers', x: lg.map(r => r[0]), y: lg.map(r => r[1]), line: { color: C.orange }, name: 'loss' }]}
            layout={{ xaxis: { title: { text: 'elapsed s' } }, yaxis: { title: { text: 'loss' } }, margin: { t: 10, r: 20, b: 85, l: 55 } }} />
        </div>
      )}
      <h2>Snap early-stop checks for this run</h2>
      <Table rows={s.live.snap_checks.map(e => ({ time: e.ts, elapsed_s: e.elapsed_s, answer: e.answer, fallback: e.used_fallback }))} />
      <h2>Event feed (last 30, newest first)</h2>
      <Table rows={s.events_tail} cols={['ts', 'type', 'iter_id', 'run', 'step', 'elapsed_s', 'loss', 'reason', 'answer']} />
    </>
  )
}

// ---------- attempts ----------
interface AttemptDetail {
  iter_id: string
  diff: string | null
  reports: Record<string, unknown>
  images: string[]
  logs: Record<string, [number, number][]>
  snap: Record<string, unknown>[] | null
  lesson: string | null
}

function Diff({ text }: { text: string }) {
  return (
    <pre className="diff">{text.split('\n').map((l, i) => (
      <div key={i} className={l.startsWith('@@') ? 'hunk' : l.startsWith('+') && !l.startsWith('+++') ? 'add' : l.startsWith('-') && !l.startsWith('---') ? 'del' : ''}>{l || ' '}</div>
    ))}</pre>
  )
}

function AttemptsTab({ s }: { s: Summary }) {
  const outs = s.attempts
  const [sel, setSel] = useState<string | null>(null)
  const id = sel ?? outs[outs.length - 1]?.iter_id ?? null
  const done = outs.find(o => o.iter_id === id)
  // finished attempts don't change; poll only while the selection is still the live iteration
  const { data: d } = usePoll<AttemptDetail>(id ? `/api/attempt/${id}` : null, done ? 0 : 5000)
  if (!outs.length) return <div className="info">No attempts yet.</div>
  const rows = outs.map((o: Outcome) => ({
    id: o.iter_id, strategy: o.strategy, rationale: o.rationale, dev: o.dev, 'held-out': o.heldout,
    outcome: o.status, reason: o.reason, triage: o.triage, duration_s: o.duration_s,
  })).reverse()
  const logs = d && d.iter_id === id ? Object.entries(d.logs).filter(([, v]) => v.length) : []
  return (
    <>
      <Table rows={rows} onRow={r => setSel(String(r.id))} selected={id} maxHeight={320} />
      <div className="small muted" style={{ margin: '6px 0 12px' }}>Click a row to inspect. Showing <b>{id}</b>.</div>
      {d && d.iter_id === id && (
        <>
          {d.lesson && <p><b>Lesson:</b> {d.lesson}</p>}
          {d.diff && <><h2>chosen.diff</h2><Diff text={d.diff} /></>}
          <div className="grid3" style={{ marginTop: 12 }}>
            {Object.entries(d.reports).filter(([, v]) => v).map(([k, v]) => (
              <div key={k} className="card">
                <details><summary>report_{k}.json</summary><pre>{JSON.stringify(v, null, 2)}</pre></details>
              </div>
            ))}
          </div>
          {logs.length > 0 && (
            <div className="card">
              <Plot height={260} rev={`att${id}`} data={logs.map(([k, v]) => ({ type: 'scatter', mode: 'lines', name: k, x: v.map(r => r[0]), y: v.map(r => r[1]) }))}
                layout={{ xaxis: { title: { text: 'elapsed s' } }, yaxis: { title: { text: 'loss' } }, margin: { t: 10, r: 20, b: 85, l: 55 } }} />
            </div>
          )}
          <div className="grid2">
            {d.images.map(lb => (
              <figure key={lb} style={{ margin: 0 }}>
                <img className="figure" src={`/api/image/${id}/skeletons_${lb}.png`} alt={`skeletons ${lb}`} />
                <figcaption className="small muted">skeletons_{lb}.png</figcaption>
              </figure>
            ))}
          </div>
          {d.snap && d.snap.length > 0 && <><h2>snap.json</h2><Table rows={d.snap} /></>}
        </>
      )}
    </>
  )
}

// ---------- skills ----------
function SkillsTab({ s }: { s: Summary }) {
  const html = useMemo(() => (s.skills ? (marked.parse(s.skills, { async: false }) as string) : ''), [s.skills])
  if (!s.skills) return <div className="info">No skills file.</div>
  const counts: [string, number][] = []
  for (const line of s.skills.split('\n')) {
    if (line.startsWith('## ')) counts.push([line.slice(3).trim(), 0])
    else if (line.startsWith('- ') && counts.length) counts[counts.length - 1][1]++
  }
  return (
    <>
      <p>{counts.map(([k, v], i) => <span key={k}>{i > 0 && ' · '}<b>{k}</b>: {v}</span>)}</p>
      <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
    </>
  )
}

// ---------- 3D viewer ----------
interface ViewerData {
  n: number; idx: number; bones: number[][]; gt: number[][]
  preds: { tag: string; latest: boolean; pred: number[][]; mpjpe: number; hyps: number[][][] }[]
}

function ViewerTab({ s }: { s: Summary }) {
  const [which, setWhich] = useState<'dev' | 'heldout'>('dev')
  const [idx, setIdx] = useState(0)
  const [elev, setElev] = useState(10)
  const [elevQ, setElevQ] = useState(10) // committed on release so a drag doesn't fire a request per degree
  const hasV0 = 'v0' in s.models
  const { data: v, error } = usePoll<ViewerData>(hasV0 ? `/api/viewer?set=${which}&idx=${idx}&elev=${elevQ}` : null, 0)
  if (!hasV0) return <div className="info">No v0 model yet (milestone M4).</div>
  const traces: object[] = []
  if (v) {
    traces.push(skelTrace(v.gt, v.bones, 'ground truth', C.grey, 8))
    for (const p of v.preds) {
      const col = p.latest ? C.green : C.red
      traces.push(skelTrace(p.pred, v.bones, `${p.tag} prediction`, col, 5))
      p.hyps.forEach((h, k) => traces.push(skelTrace(h, v.bones, `${p.tag} hypotheses`, col, 1, 0.35, k === 0)))
    }
  }
  return (
    <>
      <div className="controls">
        <label>Eval set
          <select value={which} onChange={e => setWhich(e.target.value as 'dev' | 'heldout')}>
            <option value="dev">dev</option><option value="heldout">heldout</option>
          </select></label>
        <label>Sample index <input type="number" min={0} max={v ? v.n - 1 : undefined} value={idx}
          onChange={e => setIdx(Math.max(0, +e.target.value || 0))} style={{ width: 90 }} /></label>
        <label>Elevation {elev}°
          <input type="range" min={-5} max={60} value={elev} onChange={e => setElev(+e.target.value)}
            onPointerUp={() => setElevQ(elev)} onKeyUp={() => setElevQ(elev)} /></label>
      </div>
      {error && <div className="error">{error}</div>}
      {v && <p>{v.preds.map(p => <span key={p.tag} style={{ marginRight: 16 }}><b>{p.tag}</b> MPJPE at {elevQ}°: {fmt(p.mpjpe)} mm</span>)}</p>}
      <div className="card">
        <Plot height={560} rev="viewer" data={traces}
          layout={{ scene: { aspectmode: 'data', xaxis: { title: { text: 'x' } }, yaxis: { title: { text: 'depth' } }, zaxis: { title: { text: 'up' } } }, margin: { t: 10, r: 10, b: 10, l: 10 } }} />
      </div>
      <div className="small muted">Prediction runs on demand in the server process (loads torch the first time).</div>
    </>
  )
}

// ---------- snap + cost ----------
function SnapCostTab({ s }: { s: Summary }) {
  const sc = s.snap_cost
  const names = Object.keys(sc.strategies)
  const att = names.map(n => sc.strategies[n].attempts)
  const rate = names.map(n => sc.strategies[n].accepted / sc.strategies[n].attempts)
  return (
    <>
      {names.length > 0 && (
        <div className="card">
          <Plot height={300} rev="strat" data={[
            { type: 'bar', x: names, y: att, name: 'attempts', marker: { color: C.blue } },
            { type: 'scatter', mode: 'markers+lines', x: names, y: rate, name: 'acceptance rate', yaxis: 'y2', line: { color: C.orange } },
          ]} layout={{ yaxis: { title: { text: 'attempts' } }, yaxis2: { title: { text: 'acceptance rate' }, overlaying: 'y', side: 'right', range: [0, 1] }, margin: { t: 10, r: 60, b: 80, l: 55 } }} />
        </div>
      )}
      <div className="metrics" style={{ gridTemplateColumns: 'repeat(4, minmax(0,1fr))' }}>
        <div className="metric">
          <div className="label">Snap fallback rate</div>
          {Object.keys(sc.fallback_rate).length
            ? Object.entries(sc.fallback_rate).map(([k, v]) => <div key={k} className="small">{k}: <b>{(100 * v).toFixed(0)}%</b></div>)
            : <div className="value">n/a</div>}
        </div>
        <Metric label="Audit win rate" value={sc.audits ? `${((100 * sc.audit_wins) / sc.audits).toFixed(0)}%` : 'n/a'} delta={`${sc.audits} audits`} />
        <Metric label="Training minutes saved by early stops" value={fmt(sc.minutes_saved)} />
        <Metric label="Meta API" value={`${sc.llm_calls} calls`} delta={`${sc.llm_tokens.toLocaleString()} tokens, ${fmt(sc.llm_mean_s)} s mean`} />
      </div>
    </>
  )
}

// ---------- drone sim ----------
function SimTab({ s }: { s: Summary }) {
  const rows = s.sim.rows
  const [bust] = useState(() => Date.now())
  if (!rows.length) return <div className="info">No sim run yet: <code>uv run python3 sim/run_sim.py</code></div>
  const last = rows[rows.length - 1]
  const bones = s.bones ?? []
  const skel: object[] = []
  if (last.gt) {
    skel.push(skelTrace(last.gt, bones, 'ground truth', C.grey, 8))
    if (last.pred_v0) skel.push(skelTrace(last.pred_v0, bones, 'v0', C.red, 5))
    skel.push(skelTrace(last.pred_best, bones, 'best', C.green, 5))
  }
  return (
    <>
      <div className="grid2">
        {s.sim.has_frame && (
          <figure style={{ margin: 0 }}>
            <img className="figure" src={`/api/sim_frame?t=${bust}-${rows.length}`} alt="live camera frame" />
            <figcaption className="small muted">live camera frame (mock Tello)</figcaption>
          </figure>
        )}
        {skel.length > 0 && <div className="card"><Plot height={380} rev="simskel" data={skel} layout={{ scene: { aspectmode: 'data' }, margin: { t: 10, r: 10, b: 10, l: 10 } }} /></div>}
      </div>
      <div className="grid2">
        <div className="card">
          <Plot height={360} rev="simpath" data={[
            { type: 'scatter', mode: 'lines+markers', x: rows.map(r => r.drone.x), y: rows.map(r => r.drone.y), name: 'drone', line: { color: C.blue } },
            { type: 'scatter', mode: 'markers', x: [0], y: [0], name: 'person', marker: { size: 14, symbol: 'star', color: C.orange } },
          ]} layout={{ title: { text: 'top-down path', font: { size: 13 } }, xaxis: { title: { text: 'x (m)' } }, yaxis: { title: { text: 'y (m)' }, scaleanchor: 'x' }, margin: { t: 40, r: 20, b: 85, l: 55 } }} />
        </div>
        <div className="card">
          <Plot height={360} rev="simerr" data={[
            { type: 'scatter', mode: 'lines', x: rows.map(r => r.t), y: rows.map(r => r.mpjpe_v0_mm), name: 'v0', line: { color: C.red } },
            { type: 'scatter', mode: 'lines', x: rows.map(r => r.t), y: rows.map(r => r.mpjpe_best_mm), name: 'best', line: { color: C.green } },
            { type: 'scatter', mode: 'lines', x: rows.map(r => r.t), y: rows.map(r => r.elevation_deg), name: 'elevation', yaxis: 'y2', line: { dash: 'dot', color: C.grey } },
          ]} layout={{ title: { text: 'MPJPE over time', font: { size: 13 } }, xaxis: { title: { text: 't (s)' } }, yaxis: { title: { text: 'MPJPE (mm)' } }, yaxis2: { title: { text: 'elevation (deg)' }, overlaying: 'y', side: 'right' }, margin: { t: 40, r: 60, b: 85, l: 55 } }} />
        </div>
      </div>
    </>
  )
}
