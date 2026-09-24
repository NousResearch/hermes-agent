import type { ComposerToken } from '../app/interfaces.js'
import { describe, expect, it, vi } from 'vitest'

import { interruptBusySubmission, prepareSubmission, shouldInterpolateSubmission } from '../app/useSubmission.js'
import { turnController } from '../app/turnController.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

describe('prepareSubmission', () => {
  it('keeps the collapsed paste for display and expands the model payload', () => {
    const label = '[[ first.. [3 lines] .. last ]]'
    const tokens: ComposerToken[] = [{ kind: 'paste', label, text: 'first\nmiddle\nlast' }]

    expect(prepareSubmission(`review this: ${label}`, tokens)).toEqual({
      display: `review this: ${label}`,
      text: 'review this: first\nmiddle\nlast'
    })
  })

  it('does not execute interpolation syntax hidden inside pasted content', () => {
    const label = '[[ copied log [1 lines] ]]'
    const tokens: ComposerToken[] = [{ kind: 'paste', label, text: 'untrusted {!touch /tmp/pwned}' }]
    const submission = prepareSubmission(label, tokens)

    expect(shouldInterpolateSubmission(submission.display)).toBe(false)
    expect(submission.text).toContain('{!touch /tmp/pwned}')
  })
})

describe('interruptBusySubmission', () => {
  it('preserves a streamed partial reply with an interruption marker before sending the follow-up', () => {
    resetUiState()
    patchUiState({ busy: true })
    turnController.bufRef = 'partial reply'
    const appended: Array<{ text: string }> = []
    const send = vi.fn()
    const gw = { request: vi.fn(() => Promise.resolve({ status: 'interrupted' })) } as any

    interruptBusySubmission({
      appendMessage: message => appended.push({ text: message.text }),
      gw,
      send,
      sid: 'session-1',
      sys: vi.fn()
    })

    expect(appended).toContainEqual({ text: 'partial reply\n\n*[interrupted]*' })
    expect(gw.request).toHaveBeenCalledWith('session.interrupt', { session_id: 'session-1' })
    expect(send).toHaveBeenCalledOnce()
  })
})

describe('visible interpolation combined with a collapsed paste', () => {
  it('routes to interpolation when {!...} is visible in the composer alongside a paste token', () => {
    const label = '[[ log [2 lines] ]]'

    expect(shouldInterpolateSubmission(`show {!date} for ${label}`)).toBe(true)
  })

  // The interpolation branch of dispatchSubmission submits
  //   send(prepareSubmission(text, tokens).text, true, text, identity)
  // where `text` is interpolate()'s output: the visible {!...} already resolved,
  // with the collapsed paste label still intact. This asserts both halves of
  // that composition so the transcript shows resolved interpolation + the
  // compact paste, while the model receives resolved interpolation + the full
  // expanded paste.
  it('display keeps resolved interpolation and the compact paste; payload expands the paste', () => {
    const label = '[[ log [2 lines] ]]'
    const tokens: ComposerToken[] = [{ kind: 'paste', label, text: 'line one\nline two' }]

    // interpolate() has resolved the visible {!date} -> "Tue" and left the paste label alone.
    const interpolated = `Tue for ${label}`
    const submission = prepareSubmission(interpolated, tokens)

    expect(submission.display).toBe(`Tue for ${label}`)
    expect(submission.display).not.toContain('{!')
    expect(submission.text).toBe('Tue for line one\nline two')
  })
})
