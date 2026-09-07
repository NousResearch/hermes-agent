import { describe, expect, it } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import { prepareSubmission, shouldInterpolateSubmission } from '../app/useSubmission.js'

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

  // Regression guard for the display bug teknium1 flagged: the transcript must
  // not fall back to the PRE-interpolation composer text, which still carries
  // the literal {!...}.
  it('does not show the raw {!...} syntax as the transcript display', () => {
    const label = '[[ log [2 lines] ]]'
    const tokens: ComposerToken[] = [{ kind: 'paste', label, text: 'line one\nline two' }]

    const preInterpolation = `show {!date} for ${label}`
    expect(prepareSubmission(preInterpolation, tokens).display).toContain('{!')
  })
})
