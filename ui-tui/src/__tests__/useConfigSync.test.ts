import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $uiState, resetUiState } from '../app/uiStore.js'
import { applyDisplay } from '../app/useConfigSync.js'

describe('applyDisplay', () => {
  beforeEach(() => {
    resetUiState()
  })

  it('fans every display flag out to $uiState and the bell callback', () => {
    const setBell = vi.fn()

    applyDisplay(
      {
        config: {
          display: {
            bell_on_complete: true,
            details_mode: 'expanded',
            inline_diffs: false,
            show_cost: true,
            show_reasoning: true,
            streaming: false,
            tui_compact: true,
            tui_statusbar: false
          }
        }
      },
      setBell
    )

    const s = $uiState.get()
    expect(setBell).toHaveBeenCalledWith(true)
    expect(s.compact).toBe(true)
    expect(s.detailsMode).toBe('expanded')
    expect(s.inlineDiffs).toBe(false)
    expect(s.showCost).toBe(true)
    expect(s.showReasoning).toBe(true)
    expect(s.statusBar).toBe(false)
    expect(s.streaming).toBe(false)
  })

  it('applies v1 parity defaults when display fields are missing', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: {} } }, setBell)

    const s = $uiState.get()
    expect(setBell).toHaveBeenCalledWith(false)
    expect(s.inlineDiffs).toBe(true)
    expect(s.showCost).toBe(false)
    expect(s.showReasoning).toBe(false)
    expect(s.statusBar).toBe(true)
    expect(s.streaming).toBe(true)
  })

  it('treats a null config like an empty display block', () => {
    const setBell = vi.fn()

    applyDisplay(null, setBell)

    const s = $uiState.get()
    expect(setBell).toHaveBeenCalledWith(false)
    expect(s.inlineDiffs).toBe(true)
    expect(s.streaming).toBe(true)
  })
})
