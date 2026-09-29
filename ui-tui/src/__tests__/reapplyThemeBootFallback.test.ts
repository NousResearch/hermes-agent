import { afterEach, describe, expect, it, vi } from 'vitest'

import { DEFAULT_THEME } from '../theme.js'

// #124687: the OSC-11 background probe routinely answers before the gateway
// skin event arrives, so reapplyTheme() runs with lastSkin still null. It
// must keep the boot-cached theme (already painted as frame one) instead of
// repainting the hardcoded default in between — otherwise every terminal
// that answers the probe (kitty, etc.) flashes skin -> default -> skin.
//
// But the cache is a hint, never an authority (themeBoot.ts's own contract):
// once a live signal actually DISAGREES with the cached polarity — an
// explicit `/theme` pin landing before the skin, a genuinely different OSC
// answer — that signal must win, not the frozen boot theme.
const fakeBootTheme = { ...DEFAULT_THEME, color: { ...DEFAULT_THEME.color, primary: '#123456' } }

vi.mock('../lib/themeBoot.js', () => ({
  bootSeededPin: false,
  bootTheme: fakeBootTheme,
  bootThemeIsLight: false,
  invalidateBootBackground: () => false,
  writeBootTheme: () => {}
}))

afterEach(() => {
  delete process.env.HERMES_TUI_THEME
})

describe('reapplyTheme without a skin yet', () => {
  it('falls back to the boot-cached theme when the live signal still agrees with it', async () => {
    // Mirrors a real call site (OSC-11 confirming the same polarity the
    // cache was already seeded with): the signal doesn't disagree with
    // bootThemeIsLight, so nothing new needs correcting.
    process.env.HERMES_TUI_THEME = 'dark'

    const { reapplyTheme } = await import('../app/createGatewayEventHandler.js')
    const { getUiState } = await import('../app/uiStore.js')

    reapplyTheme()

    expect(getUiState().theme.color.primary).toBe('#123456')
  })

  it('lets a disagreeing live signal override the stale boot-cached theme', async () => {
    // The cache was written dark (bootThemeIsLight: false), but a live
    // signal — here standing in for an explicit `/theme light` pin landing
    // before the skin — says light. The fresh signal must win: reusing the
    // frozen cached theme here would silently discard it for the entire
    // pre-skin window (the regression the boot-cache module's contract
    // forbids).
    process.env.HERMES_TUI_THEME = 'light'

    const { reapplyTheme } = await import('../app/createGatewayEventHandler.js')
    const { getUiState } = await import('../app/uiStore.js')

    reapplyTheme()

    expect(getUiState().theme.color.primary).not.toBe('#123456')
  })
})
