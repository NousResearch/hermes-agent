import { beforeEach, describe, expect, it } from 'vitest'

import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

// turnController.startMessage() treats "flash and yield" notices (the usage-band
// credits.usage and the one-time credits.grant_spent transition) as "show until next
// prompt": they flash once, then yield when the next turn starts. Depletion (and
// other notices) are sticky until the policy clears them.
describe('turnController.startMessage — flash-and-yield notices clear on next prompt', () => {
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

  it('clears a standing credits.grant_spent notice when a new turn starts', () => {
    // One-time "you've crossed onto top-up" heads-up — shouldn't camp the bar
    // (e.g. "Grant spent · $990 top-up left" with plenty of top-up remaining).
    patchUiState({
      notice: { key: 'credits.grant_spent', kind: 'sticky', level: 'info', text: '• Grant spent · $990.00 top-up left' }
    })
    turnController.startMessage()
    expect(getUiState().notice).toBeNull()
  })

  it('leaves a sticky credits.depleted notice across a new turn', () => {
    patchUiState({
      notice: {
        key: 'credits.depleted',
        kind: 'sticky',
        level: 'error',
        text: '✕ Credit access paused · run /credits to top up'
      }
    })
    turnController.startMessage()
    expect(getUiState().notice?.key).toBe('credits.depleted')
  })
})
