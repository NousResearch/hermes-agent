import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

describe('turnController live response text', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    resetUiState()
    turnController.fullReset()
    patchUiState({ streaming: true })
  })

  afterEach(() => {
    turnController.fullReset()
    vi.useRealTimers()
  })

  it('keeps the full response available through normal deltas and reconnect hydration', () => {
    const text = `response-start\n${'x'.repeat(16_100)}\nresponse-end`

    turnController.recordMessageDelta({ text })
    vi.runOnlyPendingTimers()
    expect(getTurnState().streaming).toBe(text)

    turnController.fullReset()
    turnController.hydrateStreamingText(text)
    expect(getTurnState().streaming).toBe(text)
  })
})
