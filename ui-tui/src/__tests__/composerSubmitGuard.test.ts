import { afterEach, describe, expect, it } from 'vitest'

import {
  hasComposerDraft,
  isComposerSubmitLocked,
  lockComposerSubmit,
  resetComposerSubmitGuard,
  shouldDrainQueuedFollowUp
} from '../app/composerSubmitGuard.js'

describe('composer submit guard', () => {
  afterEach(() => {
    resetComposerSubmitGuard()
  })

  it('locks user-enter submit after an incoming chat row', () => {
    const now = 1_000

    expect(isComposerSubmitLocked(now)).toBe(false)

    lockComposerSubmit(now, 120)

    expect(isComposerSubmitLocked(now)).toBe(true)
    expect(isComposerSubmitLocked(now + 119)).toBe(true)
    expect(isComposerSubmitLocked(now + 120)).toBe(false)
  })

  it('extends an existing lock rather than shortening it', () => {
    lockComposerSubmit(1_000, 200)
    lockComposerSubmit(1_050, 20)

    expect(isComposerSubmitLocked(1_199)).toBe(true)
    expect(isComposerSubmitLocked(1_200)).toBe(false)
  })
})

describe('hasComposerDraft', () => {
  it('treats typed text or a multiline buffer as a live draft', () => {
    expect(hasComposerDraft('', [])).toBe(false)
    expect(hasComposerDraft('   ', [])).toBe(false)
    expect(hasComposerDraft('half finished', [])).toBe(true)
    expect(hasComposerDraft('', ['line one'])).toBe(true)
  })
})

describe('shouldDrainQueuedFollowUp', () => {
  const ready = {
    busy: false,
    composerDraft: false,
    queueEdit: null,
    queueLength: 1,
    sid: 'sess-1'
  }

  it('drains when idle with a queued follow-up and an empty composer', () => {
    expect(shouldDrainQueuedFollowUp(ready)).toBe(true)
  })

  it('does not drain while the user still has text in the input', () => {
    expect(shouldDrainQueuedFollowUp({ ...ready, composerDraft: true })).toBe(false)
  })

  it('does not drain while a queued item is being edited', () => {
    expect(shouldDrainQueuedFollowUp({ ...ready, queueEdit: 0 })).toBe(false)
  })

  it('does not drain while the agent is still working', () => {
    expect(shouldDrainQueuedFollowUp({ ...ready, busy: true })).toBe(false)
  })
})
