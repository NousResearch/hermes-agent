import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

// turnController.startMessage() treats the usage-band notice (credits.usage) as
// "show until next prompt": a 50/75/90 heads-up flashes, then yields when the next
// turn starts. Depletion (and other notices) are sticky until the policy clears them.
describe('turnController.startMessage — usage-band notice clears on next prompt', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
  })

  it('clears a standing credits.usage notice when a new turn starts', () => {
    patchUiState({
      notice: { key: 'credits.usage', kind: 'sticky', level: 'warn', text: '⚠ Credits 90% used · $20.00 cap' }
    })
    turnController.startMessage()
    expect(getUiState().notice).toBeNull()
  })

  it('leaves a sticky credits.depleted notice across a new turn', () => {
    patchUiState({
      notice: { key: 'credits.depleted', kind: 'sticky', level: 'error', text: '✕ Credit access paused · run /usage for balance' }
    })
    turnController.startMessage()
    expect(getUiState().notice?.key).toBe('credits.depleted')
  })
})
