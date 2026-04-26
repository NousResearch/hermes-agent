import { describe, expect, it } from 'vitest'

import {
  freezeTurnRendering,
  getRenderableTurnState,
  patchTurnState,
  resetTurnState,
  unfreezeTurnRendering
} from '../app/turnStore.js'

describe('turn render freezing', () => {
  it('holds the render snapshot stable while live turn state keeps changing', () => {
    resetTurnState()
    patchTurnState({ streaming: 'before scroll' })
    freezeTurnRendering()

    patchTurnState({ reasoning: 'new thinking', streaming: 'new streamed text' })

    expect(getRenderableTurnState().streaming).toBe('before scroll')
    expect(getRenderableTurnState().reasoning).toBe('')

    unfreezeTurnRendering()

    expect(getRenderableTurnState().streaming).toBe('new streamed text')
    expect(getRenderableTurnState().reasoning).toBe('new thinking')
  })
})
