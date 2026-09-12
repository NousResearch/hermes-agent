import { describe, expect, it } from 'vitest'

import {
  EVENTS_MAX_RECONNECT_ATTEMPTS,
  EVENTS_RECONNECT_BASE_MS,
  EVENTS_RECONNECT_MAX_MS,
  eventsReconnectDelayMs,
  isEventsAuthRejection,
  shouldRetryEventsClose
} from './events-reconnect'

describe('eventsReconnectDelayMs', () => {
  it('doubles from the base delay', () => {
    expect(eventsReconnectDelayMs(0)).toBe(EVENTS_RECONNECT_BASE_MS)
    expect(eventsReconnectDelayMs(1)).toBe(EVENTS_RECONNECT_BASE_MS * 2)
    expect(eventsReconnectDelayMs(2)).toBe(EVENTS_RECONNECT_BASE_MS * 4)
    expect(eventsReconnectDelayMs(3)).toBe(EVENTS_RECONNECT_BASE_MS * 8)
  })

  it('clamps at the cap and never exceeds it', () => {
    for (let attempt = 0; attempt <= EVENTS_MAX_RECONNECT_ATTEMPTS + 5; attempt++) {
      expect(eventsReconnectDelayMs(attempt)).toBeLessThanOrEqual(EVENTS_RECONNECT_MAX_MS)
    }
    expect(eventsReconnectDelayMs(99)).toBe(EVENTS_RECONNECT_MAX_MS)
  })

  it('is monotonically non-decreasing', () => {
    let previous = 0
    for (let attempt = 0; attempt < 20; attempt++) {
      const delay = eventsReconnectDelayMs(attempt)
      expect(delay).toBeGreaterThanOrEqual(previous)
      previous = delay
    }
  })

  it('stays finite for absurd attempt counts', () => {
    expect(Number.isFinite(eventsReconnectDelayMs(10_000))).toBe(true)
  })
})

describe('shouldRetryEventsClose', () => {
  it('retries transient drops', () => {
    // 1005 (no status) and 1006 (abnormal) are what a killed gateway and a
    // dropped network produce respectively.
    for (const code of [1001, 1005, 1006, 1011, 1012, 1013]) {
      expect(shouldRetryEventsClose(code)).toBe(true)
    }
  })

  it('does not retry a normal closure', () => {
    expect(shouldRetryEventsClose(1000)).toBe(false)
  })

  it('does not retry auth rejections', () => {
    expect(shouldRetryEventsClose(4401)).toBe(false)
    expect(shouldRetryEventsClose(4403)).toBe(false)
  })

  it('retries when the code is missing', () => {
    expect(shouldRetryEventsClose(undefined)).toBe(true)
  })

  it('never both retries and reports an auth rejection', () => {
    for (const code of [1000, 1005, 1006, 4401, 4403, 4500]) {
      expect(shouldRetryEventsClose(code) && isEventsAuthRejection(code)).toBe(false)
    }
  })
})
