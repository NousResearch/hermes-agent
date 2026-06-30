import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

// Regression for issue #55425 — "TUI: Long responses flash — earlier content
// disappears, only tail visible".  When a long streaming response lands,
// `message.complete` must defer clearing the live streaming state until AFTER
// the committed messages are appended to the transcript.  Synchronously
// calling `idle()` first produced a frame where the live area was empty AND
// the committed message had not yet mounted, so the ScrollBox snapped to a
// smaller scrollHeight and yanked the viewport off the content the user was
// reading.  See turnController.recordMessageComplete for the full reasoning.
describe('turnController.recordMessageComplete — streaming state cleared after the append', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  afterEach(() => {
    turnController.fullReset()
  })

  it('keeps streaming state set synchronously, then clears on the next microtask', async () => {
    turnController.startMessage()
    patchUiState({ busy: true })
    turnController.hydrateStreamingText('Hello, world. Another chunk.')
    expect(getTurnState().streaming).toBe('Hello, world. Another chunk.')
    expect(getUiState().busy).toBe(true)

    const result = turnController.recordMessageComplete({ text: 'Hello, world. Another chunk.' })

    // finalMessages is still returned synchronously so the caller can append.
    expect(result.finalText).toBe('Hello, world. Another chunk.')
    expect(result.wasInterrupted).toBe(false)

    // The streaming-clear is DEFERRED — the live area should still be
    // populated for the React commit that adds the committed message, so the
    // ScrollBox's scrollHeight stays stable across the transition.
    expect(getTurnState().streaming).toBe('Hello, world. Another chunk.')
    expect(getUiState().busy).toBe(true)

    // After yielding the microtask queue, idle() has run and the live state
    // is back to the baseline.
    await Promise.resolve()
    expect(getTurnState().streaming).toBe('')
    expect(getTurnState().streamSegments).toEqual([])
    expect(getUiState().busy).toBe(false)
  })

  it('updates streaming to full payload text when boundedLiveRenderText truncated it', async () => {
    turnController.startMessage()
    patchUiState({ busy: true })
    // Simulate boundedLiveRenderText having truncated the streaming text
    // (e.g. LIVE_RENDER_MAX_CHARS = 16_000) by setting a truncated value.
    turnController.hydrateStreamingText('[showing live tail; omitted 1K lines / 12K chars]\n...final tail only')
    expect(getTurnState().streaming).toBe('[showing live tail; omitted 1K lines / 12K chars]\n...final tail only')

    // The payload text is the full, untruncated response.
    // Note: repeat(500) means the last phrase has no trailing space, so splitReasoning's
    // text.trim() produces a string equal to the original (no trailing space to strip).
    const phrase = 'The full response text that was originally long enough to be truncated.'
    const fullLongText = phrase.repeat(500)
    const result = turnController.recordMessageComplete({ text: fullLongText })

    // Synchronously, streaming is updated to the FULL payload text (not the
    // truncated version) so the streaming area height matches the committed
    // message during the transition frame. Note: splitReasoning trims the
    // result, so the stored text is fullLongText trimmed (no trailing space).
    expect(getTurnState().streaming).toBe(fullLongText.trim())
    expect(result.wasInterrupted).toBe(false)

    // Microtask still clears as usual.
    await Promise.resolve()
    expect(getTurnState().streaming).toBe('')
    expect(getTurnState().streamSegments).toEqual([])
    expect(getUiState().busy).toBe(false)
  })

  it('skips streaming update when final text is empty (reasoning-only response)', async () => {
    turnController.startMessage()
    patchUiState({ busy: true })
    turnController.hydrateStreamingText('Some previous streaming text')
    expect(getTurnState().streaming).toBe('Some previous streaming text')

    // A reasoning-only payload with no visible text.
    const result = turnController.recordMessageComplete({
      text: '',
      reasoning: 'Just thinking, no visible output'
    })

    // The streaming text should remain unchanged (no empty override).
    // The existing streaming text stays visible until the microtask clears it.
    expect(getTurnState().streaming).toBe('Some previous streaming text')
    expect(result.wasInterrupted).toBe(false)

    await Promise.resolve()
    expect(getTurnState().streaming).toBe('')
  })

  it('still cleans up synchronously when the turn was interrupted', () => {
    turnController.startMessage()
    patchUiState({ busy: true })
    turnController.hydrateStreamingText('partial response…')
    turnController.interruptTurn({
      appendMessage: () => {},
      gw: { request: () => Promise.resolve({} as never) },
      sid: 'test',
      sys: () => {}
    })

    // After interrupt, streaming is already cleared — recordMessageComplete
    // (if it even fires) should not re-trigger the microtask deferral since
    // wasInterrupted is true.
    expect(getTurnState().streaming).toBe('')
    expect(getUiState().busy).toBe(false)
  })
})
