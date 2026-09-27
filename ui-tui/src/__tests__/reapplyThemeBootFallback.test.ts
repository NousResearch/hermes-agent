import { describe, expect, it, vi } from 'vitest'

import { DEFAULT_THEME } from '../theme.js'

// #124687: the OSC-11 background probe routinely answers before the gateway
// skin event arrives, so reapplyTheme() runs with lastSkin still null. It
// must keep the boot-cached theme (already painted as frame one) instead of
// repainting the hardcoded default in between — otherwise every terminal
// that answers the probe (kitty, etc.) flashes skin -> default -> skin.
const fakeBootTheme = { ...DEFAULT_THEME, color: { ...DEFAULT_THEME.color, primary: '#123456' } }

vi.mock('../lib/themeBoot.js', () => ({
  bootSeededPin: false,
  bootTheme: fakeBootTheme,
  invalidateBootBackground: () => false,
  writeBootTheme: () => {}
}))

describe('reapplyTheme without a skin yet', () => {
  it('falls back to the boot-cached theme instead of the hardcoded default', async () => {
    const { reapplyTheme } = await import('../app/createGatewayEventHandler.js')
    const { getUiState } = await import('../app/uiStore.js')

    reapplyTheme()

    expect(getUiState().theme.color.primary).toBe('#123456')
  })
})
