import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $uiState, resetUiState } from '../app/uiStore.js'
import { applyDisplay, normalizeBusyInputMode, normalizeStatusBar } from '../app/useConfigSync.js'

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
    expect(s.statusBar).toBe('off')
    expect(s.streaming).toBe(false)
  })

  it('coerces legacy true + "on" alias to top', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: { tui_statusbar: true as unknown as 'on' } } }, setBell)
    expect($uiState.get().statusBar).toBe('top')

    applyDisplay({ config: { display: { tui_statusbar: 'on' } } }, setBell)
    expect($uiState.get().statusBar).toBe('top')
  })

  it('applies v1 parity defaults when display fields are missing', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: {} } }, setBell)

    const s = $uiState.get()
    expect(setBell).toHaveBeenCalledWith(false)
    expect(s.inlineDiffs).toBe(true)
    expect(s.showCost).toBe(false)
    expect(s.showReasoning).toBe(false)
    expect(s.statusBar).toBe('top')
    expect(s.streaming).toBe(true)
    expect(s.sections).toEqual({})
  })

  it('parses display.sections into per-section overrides', () => {
    const setBell = vi.fn()

    applyDisplay(
      {
        config: {
          display: {
            details_mode: 'collapsed',
            sections: {
              activity: 'hidden',
              tools: 'expanded',
              thinking: 'expanded',
              bogus: 'expanded'
            }
          }
        }
      },
      setBell
    )

    const s = $uiState.get()
    expect(s.detailsMode).toBe('collapsed')
    expect(s.sections).toEqual({
      activity: 'hidden',
      tools: 'expanded',
      thinking: 'expanded'
    })
  })

  it('drops invalid section modes', () => {
    const setBell = vi.fn()

    applyDisplay(
      {
        config: {
          display: {
            sections: { tools: 'maximised' as unknown as string, activity: 'hidden' }
          }
        }
      },
      setBell
    )

    expect($uiState.get().sections).toEqual({ activity: 'hidden' })
  })

  it('treats a null config like an empty display block', () => {
    const setBell = vi.fn()

    applyDisplay(null, setBell)

    const s = $uiState.get()
    expect(setBell).toHaveBeenCalledWith(false)
    expect(s.inlineDiffs).toBe(true)
    expect(s.streaming).toBe(true)
  })

  it('accepts the new string statusBar modes', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: { tui_statusbar: 'bottom' } } }, setBell)
    expect($uiState.get().statusBar).toBe('bottom')

    applyDisplay({ config: { display: { tui_statusbar: 'top' } } }, setBell)
    expect($uiState.get().statusBar).toBe('top')
  })
})

describe('normalizeStatusBar', () => {
  it('maps legacy bool + on alias to top/off', () => {
    expect(normalizeStatusBar(true)).toBe('top')
    expect(normalizeStatusBar(false)).toBe('off')
    expect(normalizeStatusBar('on')).toBe('top')
  })

  it('passes through the canonical enum', () => {
    expect(normalizeStatusBar('off')).toBe('off')
    expect(normalizeStatusBar('top')).toBe('top')
    expect(normalizeStatusBar('bottom')).toBe('bottom')
  })

  it('defaults missing/unknown values to top', () => {
    expect(normalizeStatusBar(undefined)).toBe('top')
    expect(normalizeStatusBar(null)).toBe('top')
    expect(normalizeStatusBar('sideways')).toBe('top')
    expect(normalizeStatusBar(42)).toBe('top')
  })

  it('trims whitespace and folds case', () => {
    expect(normalizeStatusBar(' Bottom ')).toBe('bottom')
    expect(normalizeStatusBar('TOP')).toBe('top')
    expect(normalizeStatusBar('  on  ')).toBe('top')
    expect(normalizeStatusBar('OFF')).toBe('off')
  })
})

describe('normalizeBusyInputMode', () => {
  it('passes through the canonical CLI parity values', () => {
    expect(normalizeBusyInputMode('queue')).toBe('queue')
    expect(normalizeBusyInputMode('steer')).toBe('steer')
    expect(normalizeBusyInputMode('interrupt')).toBe('interrupt')
  })

  it('trims and lowercases input', () => {
    expect(normalizeBusyInputMode(' Queue ')).toBe('queue')
    expect(normalizeBusyInputMode('STEER')).toBe('steer')
  })

  it('defaults to queue for missing/unknown values (TUI-only override)', () => {
    // CLI / messaging adapters keep `interrupt` as the framework default
    // (see hermes_cli/config.py + tui_gateway/server.py::_load_busy_input_mode);
    // the TUI ships `queue` because typing a follow-up while the agent
    // streams is the common authoring pattern and an unintended interrupt
    // loses work.
    expect(normalizeBusyInputMode(undefined)).toBe('queue')
    expect(normalizeBusyInputMode(null)).toBe('queue')
    expect(normalizeBusyInputMode('')).toBe('queue')
    expect(normalizeBusyInputMode('drop')).toBe('queue')
    expect(normalizeBusyInputMode(42)).toBe('queue')
  })
})

describe('applyDisplay → busy_input_mode', () => {
  beforeEach(() => {
    resetUiState()
  })

  it('threads display.busy_input_mode into $uiState', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: { busy_input_mode: 'queue' } } }, setBell)
    expect($uiState.get().busyInputMode).toBe('queue')

    applyDisplay({ config: { display: { busy_input_mode: 'steer' } } }, setBell)
    expect($uiState.get().busyInputMode).toBe('steer')
  })

  it('falls back to queue when value is missing or invalid (TUI-only default)', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: {} } }, setBell)
    expect($uiState.get().busyInputMode).toBe('queue')

    applyDisplay({ config: { display: { busy_input_mode: 'drop' } } }, setBell)
    expect($uiState.get().busyInputMode).toBe('queue')
  })
})
