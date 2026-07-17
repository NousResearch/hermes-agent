import { describe, expect, it } from 'vitest'

import type { OverlayState } from '../app/interfaces.js'
import { hasFloatingOverlay } from '../components/appLayout.js'

function overlayState(overrides: Partial<OverlayState> = {}): OverlayState {
  return {
    agents: false,
    agentsInitialHistoryIndex: 0,
    approval: null,
    billing: null,
    clarify: null,
    confirm: null,
    journey: false,
    modelPicker: false,
    pager: null,
    petPicker: false,
    pluginsHub: false,
    secret: null,
    sessions: false,
    skillsHub: false,
    sudo: null,
    ...overrides
  }
}

describe('hasFloatingOverlay', () => {
  it('returns true for blocking floating overlays rendered above the composer', () => {
    expect(hasFloatingOverlay(overlayState({ sessions: true }))).toBe(true)
    expect(hasFloatingOverlay(overlayState({ modelPicker: true }))).toBe(true)
    expect(hasFloatingOverlay(overlayState({ petPicker: true }))).toBe(true)
    expect(hasFloatingOverlay(overlayState({ skillsHub: true }))).toBe(true)
    expect(hasFloatingOverlay(overlayState({ pluginsHub: true }))).toBe(true)
    expect(hasFloatingOverlay(overlayState({ pager: { lines: ['x'], offset: 0 } }))).toBe(true)
  })

  it('returns false for non-floating overlays or an idle overlay state', () => {
    expect(hasFloatingOverlay(overlayState())).toBe(false)
    expect(
      hasFloatingOverlay(overlayState({ approval: { command: 'ls', description: 'list files' } }))
    ).toBe(false)
    expect(hasFloatingOverlay(overlayState({ clarify: { choices: null, question: 'x', requestId: 'c1' } }))).toBe(
      false
    )
    expect(
      hasFloatingOverlay(overlayState({ secret: { envVar: 'TOKEN', prompt: 'x', requestId: 's1' } }))
    ).toBe(false)
    expect(hasFloatingOverlay(overlayState({ sudo: { requestId: 'sudo1' } }))).toBe(false)
    expect(hasFloatingOverlay(overlayState({ agents: true }))).toBe(false)
  })
})
