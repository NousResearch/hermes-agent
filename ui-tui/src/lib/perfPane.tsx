// Perf instrumentation: wraps React.Profiler around named panes and writes
// commit timings to a log file when HERMES_DEV_PERF is set. Enabled per-run
// via the env var; zero-cost (Profiler is replaced by a Fragment) when off.
//
// Log format: one JSON object per line, for easy `jq` filtering. We only
// log commits that exceed a threshold (default 2ms) so the file doesn't
// fill up with sub-millisecond idle renders. Tune via HERMES_DEV_PERF_MS.
//
// Usage in consumers:
//   import { PerfPane } from './perfPane.js'
//   <PerfPane id="transcript"> ... </PerfPane>
//
// Inspect with:
//   tail -f ~/.hermes/perf.log | jq -c 'select(.actualMs > 8)'
//   jq -s 'group_by(.id) | map({id: .[0].id, count: length, p50: (sort_by(.actualMs) | .[length/2|floor].actualMs), p99: (sort_by(.actualMs) | .[length*0.99|floor].actualMs)})' ~/.hermes/perf.log

import { appendFileSync, mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

import { Profiler, type ProfilerOnRenderCallback, type ReactNode } from 'react'

const ENABLED = /^(?:1|true|yes|on)$/i.test((process.env.HERMES_DEV_PERF ?? '').trim())
const THRESHOLD_MS = Number(process.env.HERMES_DEV_PERF_MS ?? '2') || 2
const LOG_PATH = process.env.HERMES_DEV_PERF_LOG?.trim() || join(homedir(), '.hermes', 'perf.log')

let initialized = false

const ensureLogDir = () => {
  if (initialized) {
    return
  }

  initialized = true

  try {
    mkdirSync(dirname(LOG_PATH), { recursive: true })
  } catch {
    // Best-effort — if we can't create the dir (readonly fs, /tmp, etc.)
    // the appendFileSync calls below will throw silently and we drop the
    // sample. Perf logging should never crash the TUI.
  }
}

const onRender: ProfilerOnRenderCallback = (id, phase, actualMs, baseMs, startTime, commitTime) => {
  if (actualMs < THRESHOLD_MS) {
    return
  }

  ensureLogDir()

  const row = {
    actualMs: Math.round(actualMs * 100) / 100,
    baseMs: Math.round(baseMs * 100) / 100,
    commitMs: Math.round(commitTime * 100) / 100,
    id,
    phase,
    startMs: Math.round(startTime * 100) / 100,
    ts: Date.now()
  }

  try {
    appendFileSync(LOG_PATH, `${JSON.stringify(row)}\n`)
  } catch {
    // Same rationale as ensureLogDir — never crash the UI to log a sample.
  }
}

export function PerfPane({ children, id }: { children: ReactNode; id: string }) {
  if (!ENABLED) {
    return children
  }

  return (
    <Profiler id={id} onRender={onRender}>
      {children}
    </Profiler>
  )
}

export const PERF_ENABLED = ENABLED
export const PERF_LOG_PATH = LOG_PATH
