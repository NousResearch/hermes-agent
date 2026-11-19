import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

// The status bar's live t/s counter must move during the reasoning phase too.
// `reasoning.delta` and `message.delta` are separate event streams that run back
// to back inside one turn; sampling only content leaves the bar frozen on the
// previous call's average for the whole thinking phase.
describe('turnController live t/s sampling', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ liveTps: null, showReasoning: true })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const feedReasoning = (deltas: number, stepMs: number) => {
    let now = 0

    for (let i = 0; i < deltas; i++) {
      now += stepMs
      vi.setSystemTime(now)
      turnController.recordReasoningDelta('thinking ')
    }

    return now
  }

  it('sets a live rate from reasoning deltas alone, before any content delta', () => {
    turnController.startMessage()
    expect(getUiState().liveTps).toBeNull()

    feedReasoning(40, 50)

    // 40 deltas over the 2 s window -> 40 / 2 = 20 t/s.
    expect(getUiState().liveTps).toBe(20)
  })

  it('keeps sampling reasoning tokens while display.show_reasoning is off', () => {
    patchUiState({ showReasoning: false })
    turnController.startMessage()

    feedReasoning(40, 50)

    expect(getUiState().liveTps).toBe(20)
  })

  it('drops back to the server average when the stream goes quiet', () => {
    turnController.startMessage()
    feedReasoning(20, 50)
    expect(getUiState().liveTps).toBe(10)

    // Tool calls and the gap before a content stream can last seconds; a frozen
    // instantaneous rate is worse than no rate at all.
    vi.advanceTimersByTime(2300)

    expect(getUiState().liveTps).toBeNull()
  })

  it('clears the live rate when the turn starts over', () => {
    turnController.startMessage()
    feedReasoning(20, 50)
    expect(getUiState().liveTps).not.toBeNull()

    turnController.startMessage()
    expect(getUiState().liveTps).toBeNull()
  })
})
