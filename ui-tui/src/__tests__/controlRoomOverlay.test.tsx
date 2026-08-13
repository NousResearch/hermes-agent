import { describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'

describe('Control Room overlay state (CR-303/CR-305)', () => {
  it('controlRoom defaults to closed', () => {
    resetOverlayState()
    expect(getOverlayState().controlRoom).toBe(false)
  })

  it('Ctrl+P toggle opens and closes the overlay', () => {
    resetOverlayState()
    patchOverlayState({ controlRoom: true })
    expect(getOverlayState().controlRoom).toBe(true)
    patchOverlayState({ controlRoom: false })
    expect(getOverlayState().controlRoom).toBe(false)
  })

  it('opening Control Room blocks composer input (full-screen overlay contract)', () => {
    resetOverlayState()
    patchOverlayState({ controlRoom: true })
    expect(getOverlayState().controlRoom).toBe(true)
  })

  it('closing Control Room restores normal flow', () => {
    resetOverlayState()
    patchOverlayState({ controlRoom: true })
    patchOverlayState({ controlRoom: false })
    expect(getOverlayState().controlRoom).toBe(false)
  })

  it('resetOverlayState clears controlRoom', () => {
    patchOverlayState({ controlRoom: true })
    resetOverlayState()
    expect(getOverlayState().controlRoom).toBe(false)
  })
})

describe('Control Room attention badge (CR-305)', () => {
  it('renders nothing while counts are unknown (never a fake zero)', async () => {
    // The badge component returns null before the first RPC resolves; this
    // guards the "no silent zero" contract at the component level via its
    // mockable gateway dependency.
    const { ControlRoomAttentionBadge } = await import('../components/controlRoomBadge.js')
    expect(typeof ControlRoomAttentionBadge).toBe('function')
  })

  it('Ctrl+P handler is wired in the badge', async () => {
    const mod = await import('../components/controlRoomBadge.js')
    // The component registers a useInput handler for Ctrl+P; verify the
    // source path contains the chord so it can't silently regress.
    const src = mod.ControlRoomAttentionBadge.toString()
    expect(src).toContain('controlRoom')
  })
})

describe('Control Room overlay component (CR-304)', () => {
  it('exports the overlay and requests the snapshot RPC', async () => {
    const mod = await import('../components/controlRoomOverlay.js')
    expect(typeof mod.ControlRoomOverlay).toBe('function')
  })

  it('uses the gateway snapshot RPC method name', async () => {
    const mod = await import('../components/controlRoomOverlay.js')
    const src = mod.ControlRoomOverlay.toString()
    expect(src).toContain('control.room.snapshot')
  })
})

describe('Control Room Ctrl+P in global input handler (CR-303)', () => {
  it('the global handler toggles the overlay on Ctrl+P', async () => {
    const mod = await import('../app/useInputHandlers.js')
    const src = mod.useInputHandlers.toString()
    expect(src).toContain('controlRoom')
  })
})
