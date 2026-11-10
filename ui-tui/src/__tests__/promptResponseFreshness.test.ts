import { expect, it } from 'vitest'

import { capturePromptResponseGuard, patchOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

it('allows only the original prompt in the original execution to publish a response', () => {
  resetUiState()
  const info = { model: 'test', tools: {}, skills: {}, execution_epoch: 'owner', execution_generation: 1 }
  patchUiState({ sid: 'original', info })

  for (const key of ['approval', 'clarify', 'sudo', 'secret'] as const) {
    const prompt = { requestId: 'same-id' } as any
    patchOverlayState({ [key]: prompt })
    const fresh = capturePromptResponseGuard(key, prompt)
    expect(fresh()).toBe(true)
    patchUiState({ sid: 'other' })
    expect(fresh()).toBe(false)
    patchUiState({ sid: 'original', info: { ...info, execution_generation: 2 } })
    expect(fresh()).toBe(false)
    patchUiState({ info })
    patchOverlayState({ [key]: { ...prompt } })
    expect(fresh()).toBe(false)
    patchOverlayState({ [key]: null })
  }

  resetUiState()
})
