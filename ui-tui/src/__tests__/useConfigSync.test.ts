import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $uiState, resetUiState } from '../app/uiStore.js'
import {
  applyDisplay,
  hydrateFullConfig,
  normalizeBusyInputMode,
  normalizeIndicatorStyle,
  normalizeMouseTracking,
  normalizeStatusBar,
  normalizeStatusBarFields
} from '../app/useConfigSync.js'

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
            show_reasoning: true,
            streaming: false,
            tui_compact: true,
            tui_statusbar: false,
            tui_statusbar_fields: ['status', 'model', 'context', 'delegation', 'background', 'cwd']
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
    expect(s.showReasoning).toBe(true)
    expect(s.statusBar).toBe('off')
    expect(s.statusBarFields).toEqual(['status', 'model', 'context', 'delegation', 'background', 'cwd'])
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
    expect(s.showReasoning).toBe(false)
    expect(s.statusBar).toBe('top')
    expect(s.streaming).toBe(true)
    expect(s.sections).toEqual({})
  })

  it('uses documented mouse_tracking with legacy tui_mouse fallback', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: { mouse_tracking: false } } }, setBell)
    expect($uiState.get().mouseTracking).toBe('off')

    applyDisplay({ config: { display: { mouse_tracking: true, tui_mouse: false } } }, setBell)
    expect($uiState.get().mouseTracking).toBe('all')

    applyDisplay({ config: { display: { tui_mouse: false } } }, setBell)
    expect($uiState.get().mouseTracking).toBe('off')
  })

  it('threads mouse_tracking presets through to $uiState', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: { mouse_tracking: 'wheel' } } }, setBell)
    expect($uiState.get().mouseTracking).toBe('wheel')

    applyDisplay({ config: { display: { mouse_tracking: 'buttons' } } }, setBell)
    expect($uiState.get().mouseTracking).toBe('buttons')

    applyDisplay({ config: { display: { mouse_tracking: 'all' } } }, setBell)
    expect($uiState.get().mouseTracking).toBe('all')
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

describe('normalizeStatusBarFields', () => {
  it('defaults missing or invalid values to the full field set', () => {
    expect(normalizeStatusBarFields(undefined)).toContain('session_duration')
    expect(normalizeStatusBarFields('nonsense')).toContain('cost')
  })

  it('normalizes aliases, comma strings, and duplicate values', () => {
    expect(normalizeStatusBarFields('status, model, tokens, bg, path, status')).toEqual([
      'status',
      'model',
      'context',
      'background',
      'cwd'
    ])
  })

  it('ignores unsupported fields without dropping valid ones', () => {
    expect(normalizeStatusBarFields(['status', 'token-bar', 'sessions', 'bg'])).toEqual(['status', 'background'])
  })
})

describe('normalizeMouseTracking', () => {
  it('defaults to all and prefers canonical mouse_tracking over legacy tui_mouse', () => {
    expect(normalizeMouseTracking({})).toBe('all')
    expect(normalizeMouseTracking({ mouse_tracking: false })).toBe('off')
    expect(normalizeMouseTracking({ mouse_tracking: 0 })).toBe('off')
    expect(normalizeMouseTracking({ mouse_tracking: 'off' })).toBe('off')
    expect(normalizeMouseTracking({ mouse_tracking: 'false' })).toBe('off')
    expect(normalizeMouseTracking({ mouse_tracking: null, tui_mouse: false })).toBe('all')
    expect(normalizeMouseTracking({ mouse_tracking: true, tui_mouse: false })).toBe('all')
    expect(normalizeMouseTracking({ tui_mouse: false })).toBe('off')
  })

  it('accepts preset strings (wheel/buttons/all) and their aliases', () => {
    expect(normalizeMouseTracking({ mouse_tracking: 'wheel' })).toBe('wheel')
    expect(normalizeMouseTracking({ mouse_tracking: 'scroll' })).toBe('wheel')
    expect(normalizeMouseTracking({ mouse_tracking: 'buttons' })).toBe('buttons')
    expect(normalizeMouseTracking({ mouse_tracking: 'click' })).toBe('buttons')
    expect(normalizeMouseTracking({ mouse_tracking: 'all' })).toBe('all')
    expect(normalizeMouseTracking({ mouse_tracking: 'full' })).toBe('all')
    expect(normalizeMouseTracking({ mouse_tracking: 'on' })).toBe('all')
    expect(normalizeMouseTracking({ mouse_tracking: ' WHEEL ' })).toBe('wheel')
  })

  it('falls back to all for unknown strings', () => {
    expect(normalizeMouseTracking({ mouse_tracking: 'rainbows' })).toBe('all')
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

describe('normalizeIndicatorStyle', () => {
  it('passes through the canonical enum', () => {
    expect(normalizeIndicatorStyle('kaomoji')).toBe('kaomoji')
    expect(normalizeIndicatorStyle('emoji')).toBe('emoji')
    expect(normalizeIndicatorStyle('unicode')).toBe('unicode')
    expect(normalizeIndicatorStyle('ascii')).toBe('ascii')
  })

  it('trims and lowercases input', () => {
    expect(normalizeIndicatorStyle(' Emoji ')).toBe('emoji')
    expect(normalizeIndicatorStyle('UNICODE')).toBe('unicode')
  })

  it('defaults to kaomoji for missing/unknown values', () => {
    expect(normalizeIndicatorStyle(undefined)).toBe('kaomoji')
    expect(normalizeIndicatorStyle(null)).toBe('kaomoji')
    expect(normalizeIndicatorStyle('')).toBe('kaomoji')
    expect(normalizeIndicatorStyle('sparkle')).toBe('kaomoji')
    expect(normalizeIndicatorStyle(42)).toBe('kaomoji')
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

describe('applyDisplay → tui_status_indicator', () => {
  beforeEach(() => {
    resetUiState()
  })

  it('threads display.tui_status_indicator into $uiState', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: { tui_status_indicator: 'emoji' } } }, setBell)
    expect($uiState.get().indicatorStyle).toBe('emoji')

    applyDisplay({ config: { display: { tui_status_indicator: 'unicode' } } }, setBell)
    expect($uiState.get().indicatorStyle).toBe('unicode')
  })

  it('falls back to kaomoji default when missing or invalid', () => {
    const setBell = vi.fn()

    applyDisplay({ config: { display: {} } }, setBell)
    expect($uiState.get().indicatorStyle).toBe('kaomoji')

    applyDisplay({ config: { display: { tui_status_indicator: 'rainbow' } } }, setBell)
    expect($uiState.get().indicatorStyle).toBe('kaomoji')
  })
})

// Regressions from Copilot review on #19835: the config-hydration path
// for voice.record_key was untested, so a future regression in the
// hydration or mtime-reapply wiring would slip past the suite.
describe('applyDisplay → voice.record_key (#18994)', () => {
  beforeEach(() => {
    resetUiState()
  })

  it('parses voice.record_key and pushes it through the setter', () => {
    const setBell = vi.fn()
    const setVoiceRecordKey = vi.fn()

    applyDisplay({ config: { display: {}, voice: { record_key: 'ctrl+space' } } }, setBell, setVoiceRecordKey)

    expect(setVoiceRecordKey).toHaveBeenCalledWith(
      expect.objectContaining({ ch: 'space', mod: 'ctrl', named: 'space', raw: 'ctrl+space' })
    )
  })

  it('falls back to the documented default when voice.record_key is missing', () => {
    const setBell = vi.fn()
    const setVoiceRecordKey = vi.fn()

    applyDisplay({ config: { display: {} } }, setBell, setVoiceRecordKey)

    expect(setVoiceRecordKey).toHaveBeenCalledWith(expect.objectContaining({ ch: 'b', mod: 'ctrl', raw: 'ctrl+b' }))
  })

  it('is a no-op when the voice setter is not passed (back-compat)', () => {
    const setBell = vi.fn()

    // applyDisplay is used in the setVoiceEnabled-less init path too;
    // omitting the third arg must not throw.
    expect(() => applyDisplay({ config: { display: {}, voice: { record_key: 'alt+r' } } }, setBell)).not.toThrow()
  })

  it('does not reset voiceRecordKey when cfg is null (transient RPC failure)', () => {
    const setBell = vi.fn()
    const setVoiceRecordKey = vi.fn()

    // quietRpc() collapses request failures to null. Resetting the
    // cached shortcut on every null would clobber a custom binding
    // after one transient error until the next successful poll
    // (Copilot round-8 review on #19835).
    applyDisplay(null, setBell, setVoiceRecordKey)

    expect(setVoiceRecordKey).not.toHaveBeenCalled()
    // bell is still applied (defaults to false on null), so the setter
    // runs — we specifically only skip voiceRecordKey.
    expect(setBell).toHaveBeenCalledWith(false)
  })
})

// Round-12 Copilot review regression on #19835: the live mtime-reload
// path was previously untested, so a regression in the polling/RPC
// wiring to applyDisplay would only be visible at runtime. The fetch
// + apply body is now shared as ``hydrateFullConfig()``, exercised
// directly from both the initial hydration and the poll-tick body.
describe('hydrateFullConfig', () => {
  beforeEach(() => {
    resetUiState()
  })

  const makeFakeGw = (payload: unknown) =>
    ({
      request: vi.fn(() => Promise.resolve(payload)),
      on: vi.fn(),
      off: vi.fn()
    }) as any

  it('re-applies voice.record_key from a fresh config.get full response', async () => {
    const gw = makeFakeGw({ config: { display: {}, voice: { record_key: 'ctrl+o' } } })
    const setBell = vi.fn()
    const setVoiceRecordKey = vi.fn()

    await hydrateFullConfig(gw, setBell, setVoiceRecordKey)

    expect(gw.request).toHaveBeenCalledWith('config.get', { key: 'full' })
    expect(setVoiceRecordKey).toHaveBeenCalledWith(expect.objectContaining({ ch: 'o', mod: 'ctrl', raw: 'ctrl+o' }))
    expect(setBell).toHaveBeenCalledWith(false)
  })

  it('reapplies the latest value on each invocation (mtime-reload semantics)', async () => {
    const gw = makeFakeGw({ config: { display: {}, voice: { record_key: 'ctrl+b' } } })
    const setBell = vi.fn()
    const setVoiceRecordKey = vi.fn()

    await hydrateFullConfig(gw, setBell, setVoiceRecordKey)
    expect(setVoiceRecordKey).toHaveBeenLastCalledWith(expect.objectContaining({ ch: 'b' }))

    // Simulate a config edit: gw now returns a new shortcut.
    gw.request = vi.fn(() => Promise.resolve({ config: { display: {}, voice: { record_key: 'alt+space' } } }))

    await hydrateFullConfig(gw, setBell, setVoiceRecordKey)
    expect(setVoiceRecordKey).toHaveBeenLastCalledWith(
      expect.objectContaining({ ch: 'space', mod: 'alt', named: 'space' })
    )
  })

  it('leaves cached voiceRecordKey untouched when the RPC fails', async () => {
    const gw = { request: vi.fn(() => Promise.reject(new Error('boom'))), on: vi.fn(), off: vi.fn() } as any
    const setBell = vi.fn()
    const setVoiceRecordKey = vi.fn()

    const result = await hydrateFullConfig(gw, setBell, setVoiceRecordKey)

    // quietRpc() swallows the error and returns null; applyDisplay
    // sees cfg=null and skips the voice setter (Copilot round-8).
    expect(result).toBeNull()
    expect(setVoiceRecordKey).not.toHaveBeenCalled()
    // bell setter still fires — applyDisplay's null-cfg path applies
    // the documented bell default (false).
    expect(setBell).toHaveBeenCalledWith(false)
  })

  it('threads through without a voice setter (back-compat call sites)', async () => {
    const gw = makeFakeGw({ config: { display: { bell_on_complete: true } } })
    const setBell = vi.fn()

    // No third arg — applyDisplay must not throw and must still apply
    // display flags (round-2 / round-8 invariant).
    await expect(hydrateFullConfig(gw, setBell)).resolves.toBeTruthy()
    expect(setBell).toHaveBeenCalledWith(true)
  })
})
