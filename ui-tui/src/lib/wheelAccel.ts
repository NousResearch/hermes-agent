// Wheel-scroll acceleration state machine.
//
// One event = 1 row feels sluggish on trackpads (200+ ev/s) and sustained
// mouse-wheel; one event = 6 rows teleports and ruins precision.
// Heuristic on inter-event gap + direction flips:
//
//   gap < 5ms                 → same-batch burst → 1 row/event
//   gap < 40ms (native)       → ramp +0.3, cap 6
//   gap 80-500ms (xterm.js)   → mult = 1 + (mult-1)·0.5^(gap/150) + 5·decay
//                               cap 3 slow / 6 fast
//   gap > 500ms               → reset (deliberate click stays responsive)
//   flip + flip-back ≤200ms   → encoder bounce → engage wheel-mode (sticky cap)
//   5 consecutive <5ms events → trackpad flick → disengage wheel-mode
//
// Native terminals (Ghostty, iTerm2) and xterm.js embedders (VS Code,
// Cursor) emit wheel events with different cadences, hence two paths.

import { isXtermJs } from '@hermes/ink'

// ── Native (ghostty, iTerm2, WezTerm, …) ───────────────────────────────
const WHEEL_ACCEL_WINDOW_MS = 40
const WHEEL_ACCEL_STEP = 0.3
const WHEEL_ACCEL_MAX = 6

// ── Encoder bounce / wheel-mode (mechanical wheels) ────────────────────
const WHEEL_BOUNCE_GAP_MAX_MS = 200
const WHEEL_MODE_STEP = 15
const WHEEL_MODE_CAP = 15
const WHEEL_MODE_RAMP = 3
const WHEEL_MODE_IDLE_DISENGAGE_MS = 1500

// ── xterm.js (VS Code / Cursor / browser terminals) ────────────────────
const WHEEL_DECAY_HALFLIFE_MS = 150
const WHEEL_DECAY_STEP = 5
const WHEEL_BURST_MS = 5
const WHEEL_DECAY_GAP_MS = 80
const WHEEL_DECAY_CAP_SLOW = 3
const WHEEL_DECAY_CAP_FAST = 6
const WHEEL_DECAY_IDLE_MS = 500

export type WheelAccelState = {
  time: number
  mult: number
  dir: 0 | 1 | -1
  xtermJs: boolean
  /** Carried fractional scroll (xterm.js). scrollBy floors, so without
   *  this a mult of 1.5 always gives 1 row; carrying the remainder gives
   *  1,2,1,2 — correct throughput over time. */
  frac: number
  /** Native baseline rows/event. Reset on idle/reversal; ramp builds on
   *  top. xterm.js path ignores. */
  base: number
  /** Deferred direction flip (native): bounce vs reversal — next event
   *  decides. */
  pendingFlip: boolean
  /** Sticky once a flip-then-flip-back fires within the bounce window.
   *  Cleared by idle disengage or trackpad burst. */
  wheelMode: boolean
  /** Consecutive <5ms events. ≥5 → trackpad flick → disengage. */
  burstCount: number
}

export function initWheelAccel(xtermJs = false, base = 1): WheelAccelState {
  return { burstCount: 0, base, dir: 0, frac: 0, mult: base, pendingFlip: false, time: 0, wheelMode: false, xtermJs }
}

/** HERMES_TUI_SCROLL_SPEED (or CLAUDE_CODE_SCROLL_SPEED for portability).
 *  Default 1, clamped (0, 20]. */
export function readScrollSpeedBase(): number {
  const n = parseFloat(process.env.HERMES_TUI_SCROLL_SPEED ?? process.env.CLAUDE_CODE_SCROLL_SPEED ?? '')

  return Number.isFinite(n) && n > 0 ? Math.min(n, 20) : 1
}

export function initWheelAccelForHost(): WheelAccelState {
  return initWheelAccel(isXtermJs(), readScrollSpeedBase())
}

/** Compute rows for one wheel event, mutating `state`. Returns 0 when a
 *  direction flip is deferred for bounce detection — call sites should
 *  no-op on 0. */
export function computeWheelStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  return state.xtermJs ? xtermJsStep(state, dir, now) : nativeStep(state, dir, now)
}

function nativeStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  // Idle disengage runs first so a pending bounce can't mask "user paused
  // 1.5s then mouse-clicked" as a real reversal.
  if (state.wheelMode && now - state.time > WHEEL_MODE_IDLE_DISENGAGE_MS) {
    state.wheelMode = false
    state.burstCount = 0
    state.mult = state.base
  }

  if (state.pendingFlip) {
    state.pendingFlip = false

    if (dir !== state.dir || now - state.time > WHEEL_BOUNCE_GAP_MAX_MS) {
      // Real reversal (flip persisted OR flip-back too late). Commit.
      // The deferred event's 1 row is lost — acceptable latency.
      state.dir = dir
      state.time = now
      state.mult = state.base

      return Math.floor(state.mult)
    }

    state.wheelMode = true
  }

  const gap = now - state.time

  if (dir !== state.dir && state.dir !== 0) {
    state.pendingFlip = true
    state.time = now

    return 0
  }

  state.dir = dir
  state.time = now

  if (state.wheelMode) {
    if (gap < WHEEL_BURST_MS) {
      // Same-batch burst (SGR proportional) OR trackpad flick. 1 row/event;
      // trackpad flick trips the burst-count disengage.
      if (++state.burstCount >= 5) {
        state.wheelMode = false
        state.burstCount = 0
        state.mult = state.base
      } else {
        return 1
      }
    } else {
      state.burstCount = 0
    }
  }

  if (state.wheelMode) {
    const m = Math.pow(0.5, gap / WHEEL_DECAY_HALFLIFE_MS)
    const cap = Math.max(WHEEL_MODE_CAP, state.base * 2)
    const next = 1 + (state.mult - 1) * m + WHEEL_MODE_STEP * m

    state.mult = Math.min(cap, next, state.mult + WHEEL_MODE_RAMP)

    return Math.floor(state.mult)
  }

  // Trackpad / hi-res native: tight 40ms window — sub-window ramps,
  // anything slower resets to baseline.
  if (gap > WHEEL_ACCEL_WINDOW_MS) {
    state.mult = state.base
  } else {
    const cap = Math.max(WHEEL_ACCEL_MAX, state.base * 2)

    state.mult = Math.min(cap, state.mult + WHEEL_ACCEL_STEP)
  }

  return Math.floor(state.mult)
}

function xtermJsStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  const gap = now - state.time
  const sameDir = dir === state.dir

  state.time = now
  state.dir = dir

  if (sameDir && gap < WHEEL_BURST_MS) {
    return 1
  }

  if (!sameDir || gap > WHEEL_DECAY_IDLE_MS) {
    // Reversal or long idle — start at 2 so first click after a pause moves visibly.
    state.mult = 2
    state.frac = 0
  } else {
    const m = Math.pow(0.5, gap / WHEEL_DECAY_HALFLIFE_MS)
    const cap = gap >= WHEEL_DECAY_GAP_MS ? WHEEL_DECAY_CAP_SLOW : WHEEL_DECAY_CAP_FAST

    state.mult = Math.min(cap, 1 + (state.mult - 1) * m + WHEEL_DECAY_STEP * m)
  }

  const total = state.mult + state.frac
  const rows = Math.floor(total)

  state.frac = total - rows

  return rows
}
