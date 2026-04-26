// Wheel-scroll acceleration state machine.
//
// Ported from claude-code's src/components/ScrollKeybindingHandler.tsx
// (commit cb7cfba6 of their research snapshot at ~/claude-code).  The
// algorithm is theirs; the tuning constants below are theirs; this file
// is a straight port adapted to our module structure.
//
// Problem: one wheel event = 1 scrolled row feels sluggish on trackpads
// (which can fire 200+ events/sec) and during deliberate mouse scrolls.
// One wheel event = 6 rows (our old WHEEL_SCROLL_STEP=6) visually
// teleports and ruins precision.  The right answer depends on intent:
//
//   precision click  → 1 row/event
//   sustained mouse  → ramp to ~15 rows/event, decay when slowing down
//   trackpad flick   → 1 row/event per burst event (they come 100+)
//
// Heuristic: watch inter-event gaps and direction flips:
//   * gap < 5ms         → same-batch burst (SGR proportional reporting
//                         or trackpad flick) → 1 row/event
//   * gap < 40ms, same  → ramp mult by +0.3/event, cap at 6 (native path)
//   * gap < 80-500ms    → exponential decay curve (xterm.js path)
//                         mult = 1 + (mult-1)*0.5^(gap/150ms) + 5*decay
//                         capped at 3 for gaps ≥ 80ms, 6 for < 80ms
//   * gap > 500ms       → reset to 2 (deliberate click feels responsive)
//   * direction flip + bounce-back within 200ms → encoder bounce,
//                                                 engage wheel-mode
//                                                 (sticky higher cap)
//   * 5 consecutive <5ms events → trackpad flick, disengage wheel-mode
//
// Two separate paths because native terminals (Ghostty, iTerm2) and
// browser-embedded terminals (VS Code, Cursor) emit wheel events with
// different cadences.  Native sends 1 event per intended row, often
// pre-amplified at the emulator level; xterm.js sends exactly 1 event
// per notch, unamplified.

import { isXtermJs } from '@hermes/ink'

// ── Native path (ghostty, iTerm2, WezTerm, etc.) ───────────────────────
const WHEEL_ACCEL_WINDOW_MS = 40
const WHEEL_ACCEL_STEP = 0.3
const WHEEL_ACCEL_MAX = 6

// ── Encoder bounce / wheel-mode path (detected mechanical wheels) ──────
const WHEEL_BOUNCE_GAP_MAX_MS = 200
const WHEEL_MODE_STEP = 15
const WHEEL_MODE_CAP = 15
const WHEEL_MODE_RAMP = 3
const WHEEL_MODE_IDLE_DISENGAGE_MS = 1500

// ── xterm.js path (VS Code / Cursor / browser terminals) ───────────────
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
  /** Carried fractional scroll (xterm.js only). scrollBy floors, so
   *  without this a mult of 1.5 gives 1 row every time. Carrying the
   *  remainder gives 1,2,1,2 on average for mult=1.5 — correct
   *  throughput over time. */
  frac: number
  /** Native-path baseline rows/event. Reset value on idle/reversal;
   *  ramp builds on top. xterm.js path ignores this. */
  base: number
  /** Deferred direction flip (native only). Might be encoder bounce or
   *  a real reversal — resolved by the NEXT event. */
  pendingFlip: boolean
  /** Confirmed once a bounce fired (flip-then-flip-back within the
   *  bounce window).  Sticky until idle disengage or trackpad burst. */
  wheelMode: boolean
  /** Consecutive <5ms events.  Trackpad flick ≥5 → disengage wheelMode. */
  burstCount: number
}

export function initWheelAccel(xtermJs = false, base = 1): WheelAccelState {
  return {
    burstCount: 0,
    base,
    dir: 0,
    frac: 0,
    mult: base,
    pendingFlip: false,
    time: 0,
    wheelMode: false,
    xtermJs
  }
}

/** Read HERMES_TUI_SCROLL_SPEED (or CLAUDE_CODE_SCROLL_SPEED for
 *  portability from claude-code users).  Default 1, clamped (0, 20]. */
export function readScrollSpeedBase(): number {
  const raw = process.env.HERMES_TUI_SCROLL_SPEED ?? process.env.CLAUDE_CODE_SCROLL_SPEED

  if (!raw) {
    return 1
  }

  const n = parseFloat(raw)

  return Number.isNaN(n) || n <= 0 ? 1 : Math.min(n, 20)
}

/** Initialize the accel state with environment-derived defaults. */
export function initWheelAccelForHost(): WheelAccelState {
  return initWheelAccel(isXtermJs(), readScrollSpeedBase())
}

/**
 * Compute rows for one wheel event, MUTATING the accel state.  Returns 0
 * when a direction flip is deferred for bounce detection — call sites
 * should no-op on 0 (scrollBy(0) is a no-op anyway, but explicit check
 * keeps the intent obvious).
 */
export function computeWheelStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  if (!state.xtermJs) {
    return nativeStep(state, dir, now)
  }

  return xtermJsStep(state, dir, now)
}

function nativeStep(state: WheelAccelState, dir: -1 | 1, now: number): number {
  // Device-switch guard ①: idle disengage.  A pending bounce can mask
  // as a real reversal via the early return below — run this first so
  // "user stopped for 1.5s then mouse-click" restarts at baseline.
  if (state.wheelMode && now - state.time > WHEEL_MODE_IDLE_DISENGAGE_MS) {
    state.wheelMode = false
    state.burstCount = 0
    state.mult = state.base
  }

  // Resolve any deferred flip before touching state.time/dir.
  if (state.pendingFlip) {
    state.pendingFlip = false

    if (dir !== state.dir || now - state.time > WHEEL_BOUNCE_GAP_MAX_MS) {
      // Real reversal (flip persisted OR flip-back arrived too late).
      // Commit.  The deferred event's 1 row is lost (acceptable latency).
      state.dir = dir
      state.time = now
      state.mult = state.base

      return Math.floor(state.mult)
    }

    // Bounce confirmed: flipped back to original dir in the window.
    // Engage wheel-mode for sustained mouse-wheel pattern.
    state.wheelMode = true
  }

  const gap = now - state.time

  if (dir !== state.dir && state.dir !== 0) {
    // Direction flip.  Defer — next event decides bounce vs reversal.
    state.pendingFlip = true
    state.time = now

    return 0
  }

  state.dir = dir
  state.time = now

  if (state.wheelMode) {
    if (gap < WHEEL_BURST_MS) {
      // Same-batch burst (SGR proportional reporting) OR trackpad flick.
      // Give 1 row/event; trackpad flick hits the burst-count disengage.
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

  // Re-check after possible disengage above.
  if (state.wheelMode) {
    const m = Math.pow(0.5, gap / WHEEL_DECAY_HALFLIFE_MS)
    const cap = Math.max(WHEEL_MODE_CAP, state.base * 2)
    const next = 1 + (state.mult - 1) * m + WHEEL_MODE_STEP * m

    state.mult = Math.min(cap, next, state.mult + WHEEL_MODE_RAMP)

    return Math.floor(state.mult)
  }

  // Trackpad / hi-res (native, non-wheel-mode).  Tight 40ms window:
  // sub-40ms ramps, anything slower resets to baseline.
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
    // Same-batch burst — 1 row/event, same philosophy as native.
    return 1
  }

  if (!sameDir || gap > WHEEL_DECAY_IDLE_MS) {
    // Direction reversal or long idle: start at 2 so the first click
    // after a pause moves visibly.
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
