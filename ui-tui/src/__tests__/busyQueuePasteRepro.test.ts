import { describe, expect, it, vi } from 'vitest'

import type { ComposerToken } from '../app/interfaces.js'
import { submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { prepareSubmission } from '../app/useSubmission.js'
import { expandTokens } from '../domain/attachments.js'
import type { GatewayClient } from '../gatewayClient.js'
import { queueItem } from '../hooks/useQueue.js'
import { hasInterpolation, INTERPOLATION_RE } from '../protocol/interpolation.js'

describe('busy/no-session queue stores the expanded paste, not the collapsed label', () => {
  const label = '[[ log [2 lines] ]]'
  const payload = 'line one\nline two'
  const tokens: ComposerToken[] = [{ kind: 'paste', label, text: payload }]
  const full = `review this: ${label}`

  it('the queue item delivers the expanded paste at drain when live tokens are cleared', () => {
    const submission = prepareSubmission(full, tokens)
    const item = queueItem(submission.text, full)

    expect(item).toEqual({ display: `review this: ${label}`, text: `review this: ${payload}` })
    expect(expandTokens([])(item.text)).toBe(`review this: ${payload}`)
  })
})

describe('sendQueued gates shell/interpolation on the display, not the expanded text', () => {
  it('a paste whose payload begins with "!" is not shell-executed at drain', () => {
    const pasteLabel = '[[ [2 lines] ]]'
    const pastePayload = '!echo pwned\nsecond line'
    const tokens: ComposerToken[] = [{ kind: 'paste', label: pasteLabel, text: pastePayload }]
    const full = pasteLabel
    const submission = prepareSubmission(full, tokens)
    const item = queueItem(submission.text, full)

    expect(item.text.startsWith('!')).toBe(true)
    expect(item.display.startsWith('!')).toBe(false)
  })

  it('a !shell command with a collapsed paste is shell-executed with the expanded payload', () => {
    const pasteLabel = '[[ log [2 lines] ]]'
    const pastePayload = 'line one\nline two'
    const tokens: ComposerToken[] = [{ kind: 'paste', label: pasteLabel, text: pastePayload }]
    const full = `!ls ${pasteLabel}`
    const expandedArg = `!ls ${pastePayload}`
    const item = queueItem(expandedArg, full)

    expect(item.display.startsWith('!')).toBe(true)
    expect(item.text.slice(1).trim()).toBe(`ls ${pastePayload}`)
    expect(expandTokens(tokens)(item.display)).toBe(expandedArg)
  })

  it('a paste whose payload contains {!...} does not trigger interpolation at drain', () => {
    const pasteLabel = '[[ copied log [1 lines] ]]'
    const pastePayload = 'untrusted {!touch /tmp/pwned}'
    const tokens: ComposerToken[] = [{ kind: 'paste', label: pasteLabel, text: pastePayload }]
    const full = pasteLabel
    const submission = prepareSubmission(full, tokens)
    const item = queueItem(submission.text, full)

    expect(hasInterpolation(item.display)).toBe(false)
    expect(hasInterpolation(item.text)).toBe(true)
    expect(expandTokens([])(item.text)).toBe(pastePayload)
  })

  it('a visible {!...} alongside a paste does trigger interpolation at drain', () => {
    const pasteLabel = '[[ log [2 lines] ]]'
    const pastePayload = 'line one\nline two'
    const tokens: ComposerToken[] = [{ kind: 'paste', label: pasteLabel, text: pastePayload }]
    const full = `show {!date} for ${pasteLabel}`
    const submission = prepareSubmission(full, tokens)
    const item = queueItem(submission.text, full)

    expect(hasInterpolation(item.display)).toBe(true)
    expect(item.display).toBe(full)
    expect(item.text).toBe(`show {!date} for ${pastePayload}`)
  })

  // The combined collapsed-paste + visible-interpolation security hole:
  // once a visible `{!...}` opens the interpolation gate, the expanded paste
  // must not be able to smuggle in its own `{!...}` for execution. `sendQueued`
  // resolves interpolation on the display (the visible surface, where the
  // paste is still behind its `[[ … ]]` label) and re-expands the paste
  // afterwards, so a `{!...}` carried only in the pasted bytes never reaches
  // `interpolate` — it ships to the model as literal payload text.
  it('a visible {!...} does not authorize interpolation syntax smuggled in by the expanded paste', () => {
    const pasteLabel = '[[ log [1 line] ]]'
    const pastePayload = 'untrusted {!touch /tmp/pwned}'
    const tokens: ComposerToken[] = [{ kind: 'paste', label: pasteLabel, text: pastePayload }]
    const full = `show {!date} for ${pasteLabel}`
    const submission = prepareSubmission(full, tokens)
    const item = queueItem(submission.text, full, expandTokens(tokens))

    // The gate opens on the display's visible interpolation...
    expect(hasInterpolation(item.display)).toBe(true)
    // ...while the paste smuggles interpolation syntax into the expanded text.
    expect(hasInterpolation(item.text)).toBe(true)

    const executedCommands: string[] = []

    // Mirror `useSubmission.sendQueued`'s interpolation branch — `interpolate`
    // runs `shell.exec` for each `{!...}` match in the text it is given, then
    // splices the results back. Record which commands ran to pin the boundary.
    const interpolate = (text: string, then: (resolved: string) => void) => {
      const matches = [...text.matchAll(new RegExp(INTERPOLATION_RE.source, 'g'))]

      for (const m of matches) {
        executedCommands.push(m[1]!)
      }

      const resolved = matches.reduceRight(
        (acc, m) => acc.slice(0, m.index!) + 'Fri' + acc.slice(m.index! + m[0].length),
        text
      )

      then(resolved)
    }

    let submittedText = ''
    let transcriptDisplay = ''

    // sendQueued drains by interpolating the DISPLAY, then sending the
    // expanded resolved display with the resolved display as the transcript.
    interpolate(item.display, resolvedDisplay => {
      transcriptDisplay = resolvedDisplay
      submittedText = (item.expand ?? (value => value))(resolvedDisplay)
    })

    // Only the user-authored command executed; the pasted command did NOT.
    expect(executedCommands).toEqual(['date'])
    expect(executedCommands).not.toContain('touch /tmp/pwned')

    // Transcript keeps the resolved visible interpolation + the compact label.
    expect(transcriptDisplay).toBe(`show Fri for ${pasteLabel}`)
    // Model payload expands the paste; the smuggled {!...} ships as text.
    expect(submittedText).toBe(`show Fri for ${pastePayload}`)
    expect(submittedText).toContain('{!touch /tmp/pwned}')
  })
})

describe('the submit pipeline ships the expanded paste to the model and the label to the transcript', () => {
  const appendMessage = vi.fn()
  const submitted: { text: string }[] = []

  const gw = {
    request: vi.fn(async (method: string, params?: { text?: string }) => {
      if (method === 'prompt.submit' && params?.text !== undefined) {
        submitted.push({ text: params.text })
      }

      return Promise.resolve({ status: 'streaming' })
    })
  } as unknown as GatewayClient

  const deps: SubmitPromptDeps = {
    appendMessage,
    enqueue: vi.fn(),
    expand: (t: string) => t,
    gw,
    setLastUserMsg: vi.fn(),
    sys: vi.fn()
  }

  it('prompt.submit receives the expanded payload and the user bubble shows the collapsed label', async () => {
    resetUiState()
    patchUiState({ sid: 'sess-1' })
    appendMessage.mockClear()
    submitted.length = 0

    const label = '[[ log [2 lines] ]]'
    const payload = 'line one\nline two'
    const tokens: ComposerToken[] = [{ kind: 'paste', label, text: payload }]
    const full = `review this: ${label}`
    const submission = prepareSubmission(full, tokens)
    const item = queueItem(submission.text, full)

    submitPrompt(item.text, deps, true, item.display, { skipDetectDrop: true })
    await Promise.resolve()
    await Promise.resolve()

    expect(appendMessage).toHaveBeenCalledWith({ role: 'user', text: full })
    expect(submitted).toEqual([{ text: `review this: ${payload}` }])
  })
})
