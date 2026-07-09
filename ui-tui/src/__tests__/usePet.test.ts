import { describe, expect, it } from 'vitest'

import { derivePetState, isPetCellsUnsupportedError } from '../app/usePet.js'

describe('derivePetState', () => {
  it('prioritizes user-blocked state over busy/tool/reasoning states', () => {
    expect(
      derivePetState({
        awaitingInput: true,
        busy: true,
        reasoning: true,
        toolRunning: true
      })
    ).toBe('waiting')
  })

  it('maps normal agent activity to pet animation states', () => {
    expect(derivePetState({ awaitingInput: false, busy: false, reasoning: false, toolRunning: true })).toBe('run')
    expect(derivePetState({ awaitingInput: false, busy: false, reasoning: true, toolRunning: false })).toBe('review')
    expect(derivePetState({ awaitingInput: false, busy: true, reasoning: false, toolRunning: false })).toBe('run')
    expect(derivePetState({ awaitingInput: false, busy: false, reasoning: false, toolRunning: false })).toBe('idle')
  })
})

describe('isPetCellsUnsupportedError', () => {
  it('recognizes version-skew unknown-method errors for pet.cells', () => {
    expect(isPetCellsUnsupportedError(new Error('unknown method: pet.cells'))).toBe(true)
    expect(isPetCellsUnsupportedError(new Error('Unknown method:  PET.CELLS'))).toBe(true)
  })

  it('does not match unrelated RPC failures', () => {
    expect(isPetCellsUnsupportedError(new Error('unknown method: pet.gallery'))).toBe(false)
    expect(isPetCellsUnsupportedError(new Error('timeout: pet.cells'))).toBe(false)
    expect(isPetCellsUnsupportedError('unknown method: pet.cells')).toBe(false)
  })
})
