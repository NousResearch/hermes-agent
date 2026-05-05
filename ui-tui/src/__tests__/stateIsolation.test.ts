import { beforeEach, describe, expect, it } from 'vitest'

import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { $uiState, resetUiState } from '../app/uiStore.js'

const shallowEqual = <T extends Record<string, unknown>>(a: T, b: T) =>
  Object.keys(a).length === Object.keys(b).length && Object.keys(a).every(key => Object.is(a[key], b[key]))

const subscribeSelected = <T extends Record<string, unknown>>(selector: () => T) => {
  let current = selector()
  let calls = 0

  const unsubscribe = $uiState.listen(() => {
    const next = selector()

    if (shallowEqual(next, current)) {
      return
    }

    current = next
    calls++
  })

  return { calls: () => calls, unsubscribe }
}

describe('TUI state isolation', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
  })

  it('does not notify ui/composer subscribers for high-frequency turn updates', () => {
    const composerRelevant = subscribeSelected(() => ({ busy: $uiState.get().busy, sid: $uiState.get().sid }))

    try {
      for (let i = 0; i < 50; i++) {
        patchTurnState({ streaming: `token ${i}` })
      }
    } finally {
      composerRelevant.unsubscribe()
    }

    expect(composerRelevant.calls()).toBe(0)
  })
})
