import { describe, expect, it } from 'vitest'

import { computePrecisionWheelStep, initPrecisionWheel } from '../lib/precisionWheel.js'

describe('precisionWheel', () => {
  it('passes the first modifier-held wheel event', () => {
    const s = initPrecisionWheel()

    expect(computePrecisionWheelStep(s, 1, true, 1000)).toEqual({ active: true, entered: true, rows: 1 })
  })

  it('coalesces same-frame events without throttling line-by-line scroll', () => {
    const s = initPrecisionWheel()

    computePrecisionWheelStep(s, 1, true, 1000)

    expect(computePrecisionWheelStep(s, 1, true, 1008).rows).toBe(0)
    expect(computePrecisionWheelStep(s, 1, true, 1016).rows).toBe(1)
  })

  it('keeps queued momentum in precision mode briefly after modifier release', () => {
    const s = initPrecisionWheel()

    computePrecisionWheelStep(s, 1, true, 1000)

    expect(computePrecisionWheelStep(s, 1, false, 1050)).toMatchObject({ active: true, rows: 1 })
  })

  it('leaves precision mode once modifier-free momentum goes idle', () => {
    const s = initPrecisionWheel()

    computePrecisionWheelStep(s, 1, true, 1000)

    expect(computePrecisionWheelStep(s, 1, false, 1100)).toEqual({ active: false, entered: false, rows: 0 })
  })

  it('does not coalesce immediate reversals', () => {
    const s = initPrecisionWheel()

    computePrecisionWheelStep(s, 1, true, 1000)

    expect(computePrecisionWheelStep(s, -1, true, 1008).rows).toBe(1)
  })
})
