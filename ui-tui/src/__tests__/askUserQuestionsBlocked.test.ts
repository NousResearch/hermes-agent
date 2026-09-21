import { afterEach, describe, expect, it } from 'vitest'

import { $isBlocked, getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'

describe('askUserQuestions is a blocking overlay', () => {
  afterEach(() => {
    resetOverlayState()
  })

  it('does not block when no AUQ prompt is open', () => {
    expect(getOverlayState().askUserQuestions).toBeNull()
    expect($isBlocked.get()).toBe(false)
  })

  it('blocks the composer while an AUQ prompt is open', () => {
    patchOverlayState({
      askUserQuestions: {
        questions: [
          {
            header: 'SCOPE',
            multiSelect: false,
            options: [{ label: 'Full' }],
            question: 'Scope?'
          }
        ],
        requestId: 'auq-1'
      }
    })

    expect($isBlocked.get()).toBe(true)
  })
})