const PRECISION_WHEEL_FRAME_MS = 16
const PRECISION_WHEEL_STICKY_MS = 80

export type PrecisionWheelState = {
  active: boolean
  dir: 0 | -1 | 1
  lastEventAtMs: number
  lastScrollAtMs: number
}

export type PrecisionWheelStep = {
  active: boolean
  entered: boolean
  rows: 0 | 1
}

export function initPrecisionWheel(): PrecisionWheelState {
  return { active: false, dir: 0, lastEventAtMs: 0, lastScrollAtMs: 0 }
}

export function computePrecisionWheelStep(
  state: PrecisionWheelState,
  dir: -1 | 1,
  hasModifier: boolean,
  now: number
): PrecisionWheelStep {
  const active = hasModifier || now - state.lastEventAtMs < PRECISION_WHEEL_STICKY_MS

  if (!active) {
    state.active = false

    return { active: false, entered: false, rows: 0 }
  }

  const entered = !state.active

  state.active = true
  state.lastEventAtMs = now

  if (dir === state.dir && now - state.lastScrollAtMs < PRECISION_WHEEL_FRAME_MS) {
    return { active: true, entered, rows: 0 }
  }

  state.dir = dir
  state.lastScrollAtMs = now

  return { active: true, entered, rows: 1 }
}
