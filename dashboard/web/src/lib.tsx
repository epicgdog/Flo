import Plotly from 'plotly.js-dist-min'
import { useEffect, useRef, useState } from 'react'

// ---------- API types (mirror dashboard/server.py) ----------
export type Status = 'accepted' | 'rejected' | 'killed_early' | 'failed'
export interface Outcome {
  iter_id: string
  strategy?: string
  rationale?: string
  accepted?: boolean
  reason?: string | null
  tag?: string | null
  dev?: number | null
  dev2?: number | null
  heldout?: number | null
  best_dev_before?: number | null
  best_heldout_before?: number | null
  triage?: unknown
  lesson?: string
  duration_s?: number
  status: Status
}
export interface RunState {
  run_id?: string
  started_at?: string
  ends_at?: string
  iter_id?: string
  phase?: string
  strategy?: string
  rationale?: string
  train_elapsed_s?: number
  train_loss?: number | null
  best?: { tag?: string; dev?: number; heldout?: number; iter_id?: string }
  counts?: { iters?: number; accepted?: number; rejected?: number; killed_early?: number; errors?: number }
  last_update?: string
}
export type Log = [number, number][]
export type Ev = Record<string, unknown> & { ts?: string; type?: string; iter_id?: string }
export interface SimRow {
  t: number
  drone: { x: number; y: number }
  gt?: number[][]
  pred_v0?: number[][]
  pred_best: number[][]
  mpjpe_v0_mm?: number
  mpjpe_best_mm?: number
  elevation_deg?: number
}
export interface Summary {
  server_time: string
  state: RunState
  baseline: { dev?: number; heldout?: number }
  base_outcome: Outcome | null
  attempts: Outcome[]
  loss_logs: Record<string, Log>
  live: { label?: string | null; log: Log; snap_checks: Ev[] }
  iter_end: Record<string, string>
  events_tail: Ev[]
  budget_s: number
  skills: string | null
  snap_cost: {
    strategies: Record<string, { attempts: number; accepted: number }>
    fallback_rate: Record<string, number>
    audits: number
    audit_wins: number
    minutes_saved: number
    llm_calls: number
    llm_tokens: number
    llm_mean_s: number
  }
  sim: { rows: SimRow[]; has_frame: boolean }
  models: Record<string, string>
  bones: number[][]
}

// ---------- helpers ----------
export const fmt = (v: unknown, d = 1) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '–')
export const parseTs = (s?: string) => (s ? new Date(s) : null)

export function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}
export const C = {
  get green() { return cssVar('--green') },
  get red() { return cssVar('--red') },
  get grey() { return cssVar('--grey') },
  get ink() { return cssVar('--ink') },
  get blue() { return cssVar('--blue') },
  get orange() { return cssVar('--orange') },
}
export const statusColor = (s: Status) =>
  ({ accepted: C.green, rejected: C.red, killed_early: C.grey, failed: C.ink })[s]

export function usePoll<T>(url: string | null, everyMs: number) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!url) return
    let alive = true
    let timer: number | undefined
    const tick = async () => {
      try {
        const r = await fetch(url)
        const j = await r.json()
        if (!alive) return
        if (!r.ok || j.error) setError(j.error || `HTTP ${r.status}`)
        else { setData(j); setError(null) }
      } catch (e) {
        if (alive) setError(String(e))
      }
      if (alive && everyMs > 0) timer = window.setTimeout(tick, everyMs)
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [url, everyMs])
  return { data, error }
}

// ---------- Plotly wrapper ----------
// uirevision keeps zoom/legend toggles across the 5 s refreshes.
type AnyObj = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
export function Plot({ data, layout, height = 340, rev = 'x' }: { data: AnyObj[]; layout?: AnyObj; height?: number; rev?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [dark, setDark] = useState(() => matchMedia('(prefers-color-scheme: dark)').matches)
  useEffect(() => {
    const m = matchMedia('(prefers-color-scheme: dark)')
    const f = () => setDark(m.matches)
    m.addEventListener('change', f)
    return () => m.removeEventListener('change', f)
  }, [])
  useEffect(() => {
    if (!ref.current) return
    const fg = cssVar('--fg'), grid = cssVar('--grid'), muted = cssVar('--muted')
    const axis = { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: muted } }
    const base: AnyObj = {
      height, autosize: true, uirevision: rev,
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: fg, size: 12 }, margin: { t: 30, r: 50, b: 40, l: 55 },
      legend: { orientation: 'h', yanchor: 'top', y: -0.25 }, xaxis: axis, yaxis: axis,
      hoverlabel: { font: { size: 12 } },
    }
    const l = { ...base, ...layout }
    for (const k of ['xaxis', 'yaxis', 'yaxis2']) if (layout?.[k]) l[k] = { ...axis, ...layout[k] }
    if (layout?.scene) {
      const sa = { ...axis, backgroundcolor: 'rgba(0,0,0,0)', showbackground: false }
      l.scene = { ...layout.scene, xaxis: { ...sa, ...layout.scene.xaxis }, yaxis: { ...sa, ...layout.scene.yaxis }, zaxis: { ...sa, ...layout.scene.zaxis } }
    }
    Plotly.react(ref.current, data as never, l as never, { responsive: true, displaylogo: false })
  }, [data, layout, height, rev, dark])
  useEffect(() => {
    const el = ref.current
    return () => { if (el) Plotly.purge(el) }
  }, [])
  return <div ref={ref} style={{ width: '100%', minHeight: height }} />
}

// ---------- Table ----------
export function Table({ rows, cols, onRow, selected, maxHeight }: {
  rows: Record<string, unknown>[]
  cols?: string[]
  onRow?: (r: Record<string, unknown>) => void
  selected?: unknown
  maxHeight?: number
}) {
  if (!rows.length) return <div className="info">No rows.</div>
  const keys = cols ?? Array.from(new Set(rows.flatMap(r => Object.keys(r))))
  const cell = (v: unknown) => {
    if (v == null) return ''
    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3)
    if (typeof v === 'object') return JSON.stringify(v)
    return String(v)
  }
  return (
    <div className="tablewrap" style={maxHeight ? { maxHeight } : undefined}>
      <table>
        <thead><tr>{keys.map(k => <th key={k}>{k}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={(onRow ? 'clickable ' : '') + (selected != null && r.id === selected ? 'selected' : '')}
              onClick={onRow ? () => onRow(r) : undefined}>
              {keys.map(k => {
                const v = r[k]
                if (k === 'outcome' && typeof v === 'string') return <td key={k}><span className={`pill ${v}`}>{v}</span></td>
                return <td key={k} className={typeof v === 'number' ? 'num' : undefined}>{cell(v)}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------- Skeleton (3D) ----------
// Camera coords -> plot: x, depth (z), up (-y). Same mapping as the Streamlit viewer.
export function skelTrace(p: number[][], bones: number[][], name: string, color: string, width: number, opacity = 1, showlegend = true) {
  const xs: (number | null)[] = [], ys: (number | null)[] = [], zs: (number | null)[] = []
  for (const [a, b] of bones) {
    xs.push(p[a][0], p[b][0], null)
    ys.push(p[a][2], p[b][2], null)
    zs.push(-p[a][1], -p[b][1], null)
  }
  return { type: 'scatter3d', mode: 'lines', x: xs, y: ys, z: zs, name, showlegend, legendgroup: name, line: { color, width }, opacity }
}
