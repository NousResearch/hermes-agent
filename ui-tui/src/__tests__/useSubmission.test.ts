import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { turnController } from '../app/turnController.js'
import { promptAcceptedWithoutEvents } from '../app/useSubmission.js'

describe('promptAcceptedWithoutEvents', () => {
  it('detects an accepted prompt that produced no stream events while still busy', () => {
    expect(promptAcceptedWithoutEvents({ busy: true, sid: 's1' }, 's1', 7, 7)).toBe(true)
  })

  it('does not fire after stream events, session changes, or busy clears', () => {
    expect(promptAcceptedWithoutEvents({ busy: true, sid: 's1' }, 's1', 7, 8)).toBe(false)
    expect(promptAcceptedWithoutEvents({ busy: true, sid: 's2' }, 's1', 7, 7)).toBe(false)
    expect(promptAcceptedWithoutEvents({ busy: false, sid: 's1' }, 's1', 7, 7)).toBe(false)
  })
})

describe('prompt acceptance watchdog ownership', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    turnController.reset()
  })

  afterEach(() => {
    turnController.reset()
    vi.useRealTimers()
  })

  it('cancels an interrupted submission watchdog without touching the newer prompt', () => {
    const fired: string[] = []
    const oldGeneration = turnController.resetForSubmit()

    expect(turnController.acceptPromptSubmission(oldGeneration, 60_000, () => fired.push('old'))).toBe(true)
    vi.advanceTimersByTime(30_000)

    turnController.interruptTurn({
      appendMessage: vi.fn(),
      gw: { request: vi.fn(() => Promise.resolve({})) },
      sid: 's1',
      sys: vi.fn()
    })

    const newGeneration = turnController.resetForSubmit()

    expect(turnController.acceptPromptSubmission(newGeneration, 60_000, () => fired.push('new'))).toBe(true)

    // The old deadline arrives while the newer prompt still owns busy state.
    vi.advanceTimersByTime(30_000)
    expect(fired).toEqual([])

    vi.advanceTimersByTime(30_000)

    expect(fired).toEqual(['new'])
  })

  it('ignores a late older acceptance without cancelling the newer watchdog', () => {
    const fired: string[] = []
    const oldGeneration = turnController.resetForSubmit()
    const newGeneration = turnController.resetForSubmit()

    expect(turnController.acceptPromptSubmission(newGeneration, 60_000, () => fired.push('new'))).toBe(true)
    expect(turnController.acceptPromptSubmission(oldGeneration, 60_000, () => fired.push('old'))).toBe(false)

    vi.advanceTimersByTime(60_000)

    expect(fired).toEqual(['new'])
  })
})
