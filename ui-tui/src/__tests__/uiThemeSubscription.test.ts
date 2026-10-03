import { afterEach, describe, expect, it } from 'vitest'

import { $uiState, $uiTheme, getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

afterEach(resetUiState)

describe('ui theme subscription', () => {
  it('does not notify theme-only consumers for unrelated ui-state churn', () => {
    let fullStoreWakeups = 0
    let themeWakeups = 0
    const stopFull = $uiState.listen(() => {
      fullStoreWakeups += 1
    })
    const stopTheme = $uiTheme.listen(() => {
      themeWakeups += 1
    })

    try {
      const fullBaseline = fullStoreWakeups
      const themeBaseline = themeWakeups

      for (let i = 0; i < 20; i++) {
        patchUiState({ status: `streaming-${i}` })
      }

      expect(fullStoreWakeups - fullBaseline).toBe(20)
      expect(themeWakeups - themeBaseline).toBe(0)

      patchUiState({ theme: { ...getUiState().theme } })
      expect(themeWakeups - themeBaseline).toBe(1)
    } finally {
      stopTheme()
      stopFull()
    }
  })
})
