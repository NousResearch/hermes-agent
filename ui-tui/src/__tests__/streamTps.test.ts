import { describe, expect, it } from 'vitest'

import { createStreamTpsCounter } from '../app/streamTps.js'

describe('createStreamTpsCounter', () => {
  it('needs three samples before it reports, then emits once per second', () => {
    const counter = createStreamTpsCounter()

    // First call passes the throttle but holds a single sample.
    expect(counter.accumulate(0)).toBeUndefined()
    expect(counter.accumulate(200)).toBeUndefined()
    expect(counter.accumulate(400)).toBeUndefined()
    // Under the 1 s throttle: nothing, even with enough samples.
    expect(counter.accumulate(900)).toBeUndefined()

    // Five deltas inside the 2 s window -> 5 / 2 = 2.5 t/s.
    expect(counter.accumulate(1000)).toBe(2.5)
  })

  it('drops samples older than the rolling window', () => {
    const counter = createStreamTpsCounter()

    counter.accumulate(0)
    counter.accumulate(100)
    counter.accumulate(200)
    // Four samples in the window by now (0, 100, 200 and this one) -> 2 t/s.
    expect(counter.accumulate(1100)).toBe(2)

    // Throttled: the previous emit was 400 ms ago.
    expect(counter.accumulate(1500)).toBeUndefined()

    // 3.2 s later every earlier sample is older than 2 s; one sample is not a rate.
    expect(counter.accumulate(3200)).toBeUndefined()
    expect(counter.accumulate(3300)).toBeUndefined()
    expect(counter.accumulate(3400)).toBeUndefined()
    // 3200, 3300, 3400 and this one survive the window -> 4 / 2 = 2 t/s.
    expect(counter.accumulate(4400)).toBe(2)
  })

  it('restarts from scratch after reset', () => {
    const counter = createStreamTpsCounter()

    counter.accumulate(0)
    counter.accumulate(500)
    expect(counter.accumulate(1000)).toBe(1.5)

    counter.reset()

    // A new turn starts with one sample, then the throttle mutes the burst.
    expect(counter.accumulate(2000)).toBeUndefined()
    expect(counter.accumulate(2500)).toBeUndefined()
    expect(counter.accumulate(2600)).toBeUndefined()
    // Four samples in the window again.
    expect(counter.accumulate(3000)).toBe(2)
  })
})
