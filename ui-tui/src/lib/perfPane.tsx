// Perf instrumentation for the full render pipeline.
//
// Two sources of timing:
//   1. React.Profiler wrapper (PerfPane) → per-pane commit times. Shows
//      which subtree is reconciling and for how long.
//   2. Ink onFrame callback (logFrameEvent) → per-frame pipeline phases:
//      yoga (calculateLayout), renderer (DOM → screen buffer), diff
//      (prev vs current screen → patches), optimize (patch merge/dedupe),
//      write (serialize → ANSI → stdout), plus yoga counters (visited,
//      measured, cacheHits, live). Shows where the time goes BELOW React.
//
// Both sources gate on HERMES_DEV_PERF=1 and dump JSON-lines to the same
// log (default ~/.hermes/perf.log, override via HERMES_DEV_PERF_LOG).
// Events are tagged { src: 'react' | 'frame' } so jq can split them.
//
// Threshold HERMES_DEV_PERF_MS (default 2ms) skips sub-millisecond idle
// frames. For the 2fps-during-PageUp investigation, set
// HERMES_DEV_PERF_MS=0 to capture everything, then filter with jq.
//
// Zero cost when the env var is unset: PerfPane returns children
// directly (no Profiler fiber), logFrameEvent is a noop on the onFrame
// callback — the ink instance isn't given the callback at all.
//
// Usage:
//   # entry.tsx wires logFrameEvent into render()
//   import { logFrameEvent, PerfPane } from './lib/perfPane.js'
//   render(<App/>, { onFrame: logFrameEvent })
//
// Analysis helpers (once you've captured a session):
//   tail -f ~/.hermes/perf.log | jq -c 'select(.src=="frame" and .durationMs > 16)'
//   # p50/p99 per phase across frame events:
//   jq -s '[.[] | select(.src=="frame")] |
//     {n: length,
//      dur_p50: (sort_by(.durationMs) | .[length/2|floor].durationMs),
//      dur_p99: (sort_by(.durationMs) | .[length*0.99|floor].durationMs),
//      yoga_p99: (sort_by(.phases.yoga) | .[length*0.99|floor].phases.yoga),
//      write_p99: (sort_by(.phases.write) | .[length*0.99|floor].phases.write),
//      diff_p99: (sort_by(.phases.diff) | .[length*0.99|floor].phases.diff),
//      patches_p99: (sort_by(.phases.patches) | .[length*0.99|floor].phases.patches)}' \
//     ~/.hermes/perf.log

import { appendFileSync, mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

import type { FrameEvent } from '@hermes/ink'
import { Profiler, type ProfilerOnRenderCallback, type ReactNode } from 'react'

const ENABLED = /^(?:1|true|yes|on)$/i.test((process.env.HERMES_DEV_PERF ?? '').trim())
const THRESHOLD_MS = Number(process.env.HERMES_DEV_PERF_MS ?? '2') || 0
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

const writeRow = (row: Record<string, unknown>) => {
  ensureLogDir()

  try {
    appendFileSync(LOG_PATH, `${JSON.stringify(row)}\n`)
  } catch {
    // Same rationale as ensureLogDir — never crash the UI to log a sample.
  }
}

const round2 = (n: number) => Math.round(n * 100) / 100

const onRender: ProfilerOnRenderCallback = (id, phase, actualMs, baseMs, startTime, commitTime) => {
  if (actualMs < THRESHOLD_MS) {
    return
  }

  writeRow({
    actualMs: round2(actualMs),
    baseMs: round2(baseMs),
    commitMs: round2(commitTime),
    id,
    phase,
    src: 'react',
    startMs: round2(startTime),
    ts: Date.now()
  })
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

/**
 * Ink onFrame handler. Captures the FULL render pipeline: yoga calculateLayout,
 * DOM → screen buffer, screen diff, patch optimize, and stdout write.
 *
 * Returns `undefined` when disabled so `render()` doesn't attach the callback —
 * ink only pays the timing cost when the callback is truthy.
 */
export const logFrameEvent = ENABLED
  ? (event: FrameEvent) => {
      if (event.durationMs < THRESHOLD_MS) {
        return
      }

      writeRow({
        durationMs: round2(event.durationMs),
        flickers: event.flickers.length ? event.flickers : undefined,
        phases: event.phases
          ? {
              commit: round2(event.phases.commit),
              diff: round2(event.phases.diff),
              optimize: round2(event.phases.optimize),
              patches: event.phases.patches,
              renderer: round2(event.phases.renderer),
              write: round2(event.phases.write),
              yoga: round2(event.phases.yoga),
              yogaCacheHits: event.phases.yogaCacheHits,
              yogaLive: event.phases.yogaLive,
              yogaMeasured: event.phases.yogaMeasured,
              yogaVisited: event.phases.yogaVisited
            }
          : undefined,
        src: 'frame',
        ts: Date.now()
      })
    }
  : undefined

export const PERF_ENABLED = ENABLED
export const PERF_LOG_PATH = LOG_PATH
