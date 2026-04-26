import { afterEach, describe, expect, it, vi } from 'vitest'

import { getInteractionMode, markScrolling, markTyping, resetInteractionMode } from '../app/interactionMode.js'
import { SCROLLING_IDLE_MS, TYPING_IDLE_MS } from '../config/timing.js'

describe('interactionMode', () => {
  afterEach(() => {
    resetInteractionMode()
    vi.useRealTimers()
  })

  it('holds scrolling mode briefly then returns idle', () => {
    vi.useFakeTimers()
    markScrolling()
    expect(getInteractionMode()).toBe('scrolling')
    vi.advanceTimersByTime(SCROLLING_IDLE_MS)
    expect(getInteractionMode()).toBe('idle')
  })

  it('typing takes priority over scrolling', () => {
    vi.useFakeTimers()
    markTyping()
    markScrolling()
    expect(getInteractionMode()).toBe('typing')
    vi.advanceTimersByTime(TYPING_IDLE_MS)
    expect(getInteractionMode()).toBe('idle')
  })
})
