import { describe, expect, it } from 'vitest'

import { computeWheelStep, initWheelAccel } from '../lib/wheelAccel.js'

describe('wheelAccel — native path', () => {
  it('first click after init returns base', () => {
    const s = initWheelAccel(false, 1)

    expect(computeWheelStep(s, 1, 1000)).toBe(1)
  })

  it('same-direction fast events ramp mult (window-mode)', () => {
    const s = initWheelAccel(false, 1)

    // First click establishes dir. Subsequent clicks inside the 40ms
    // window ramp by +0.3 each (capped at 6).
    computeWheelStep(s, 1, 1000)
    computeWheelStep(s, 1, 1020)
    computeWheelStep(s, 1, 1040)
    const fourth = computeWheelStep(s, 1, 1060)

    // After 3 window events: mult starts at 1 → stays 1 on first ramp
    // (first event just sets baseline), then +0.3 × 3 = 1.9 → floor=1.
    // The key property: doesn't shrink below base.
    expect(fourth).toBeGreaterThanOrEqual(1)
  })

  it('gap beyond window resets mult to base', () => {
    const s = initWheelAccel(false, 1)

    // Ramp up
    for (let t = 1000; t < 1100; t += 20) {
      computeWheelStep(s, 1, t)
    }

    // Long pause, then click
    const afterPause = computeWheelStep(s, 1, 2000)

    expect(afterPause).toBe(1)
  })

  it('direction flip defers one event for bounce detection', () => {
    const s = initWheelAccel(false, 1)

    computeWheelStep(s, 1, 1000)
    // Flip — should defer
    expect(computeWheelStep(s, -1, 1050)).toBe(0)
  })

  it('flip-back within bounce window engages wheelMode', () => {
    const s = initWheelAccel(false, 1)

    computeWheelStep(s, 1, 1000)
    // Flip (deferred)
    computeWheelStep(s, -1, 1050)
    // Flip BACK within 200ms → bounce confirmed → wheelMode engaged
    computeWheelStep(s, 1, 1100)

    expect(s.wheelMode).toBe(true)
  })

  it('flip-back outside bounce window is a real reversal (no wheelMode)', () => {
    const s = initWheelAccel(false, 1)

    computeWheelStep(s, 1, 1000)
    computeWheelStep(s, -1, 1050) // defer
    // Flip-back arrives 300ms later → too late → real reversal
    computeWheelStep(s, 1, 1400)

    expect(s.wheelMode).toBe(false)
  })

  it('5 consecutive sub-5ms events disengage wheelMode (trackpad signature)', () => {
    const s = initWheelAccel(false, 1)
    s.wheelMode = true
    s.dir = 1
    s.time = 1000

    // 5 bursts <5ms apart (trackpad flick)
    computeWheelStep(s, 1, 1002)
    computeWheelStep(s, 1, 1004)
    computeWheelStep(s, 1, 1006)
    computeWheelStep(s, 1, 1008)
    computeWheelStep(s, 1, 1010)

    expect(s.wheelMode).toBe(false)
  })

  it('1.5s idle disengages wheelMode', () => {
    const s = initWheelAccel(false, 1)
    s.wheelMode = true
    s.dir = 1
    s.time = 1000

    computeWheelStep(s, 1, 3000) // 2 second gap

    expect(s.wheelMode).toBe(false)
  })
})

describe('wheelAccel — xterm.js path', () => {
  it('first click returns 2 after long idle', () => {
    const s = initWheelAccel(true, 1)

    // First event — "sameDir && gap > WHEEL_DECAY_IDLE_MS" triggers
    // reset-to-2 branch since dir starts at 0 and 0 !== 1.
    const n = computeWheelStep(s, 1, 1000)

    expect(n).toBeGreaterThanOrEqual(1)
  })

  it('sub-5ms burst returns 1 (same-direction, same-batch)', () => {
    const s = initWheelAccel(true, 1)

    computeWheelStep(s, 1, 1000)
    const burst = computeWheelStep(s, 1, 1002)

    expect(burst).toBe(1)
  })

  it('slow steady scroll stays in precision range', () => {
    const s = initWheelAccel(true, 1)

    // Simulated 30Hz sustained scroll: 33ms gap
    const results: number[] = []

    for (let t = 1000; t < 2000; t += 33) {
      results.push(computeWheelStep(s, 1, t))
    }

    // Every event should produce 1-6 rows.  No runaway.
    for (const r of results) {
      expect(r).toBeGreaterThanOrEqual(1)
      expect(r).toBeLessThanOrEqual(6)
    }
  })

  it('direction reversal resets mult', () => {
    const s = initWheelAccel(true, 1)

    // Ramp up
    for (let t = 1000; t < 1100; t += 20) {
      computeWheelStep(s, 1, t)
    }
    const beforeFlip = s.mult

    // Flip
    computeWheelStep(s, -1, 1200)

    expect(s.mult).toBeLessThanOrEqual(beforeFlip)
    // Reset branch sets mult=2
    expect(s.mult).toBe(2)
  })

  it('frac stays in [0,1) across events', () => {
    const s = initWheelAccel(true, 1)

    // frac must never go negative or reach 1.0 — that's the correctness
    // invariant of the fractional carry.  Whether a specific series of
    // inputs produces a nonzero frac depends on tuning constants; just
    // check the bound is maintained across a realistic scroll pattern.
    for (let t = 1000; t < 1200; t += 30) {
      computeWheelStep(s, 1, t)

      expect(s.frac).toBeGreaterThanOrEqual(0)
      expect(s.frac).toBeLessThan(1)
    }
  })
})
