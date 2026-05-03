import { describe, expect, it } from 'vitest'

import { classifyKeyEvent } from '../lib/terminalShortcuts.js'
import type { TerminalCapabilities } from '../lib/terminalCapabilities.js'

const baseCaps: TerminalCapabilities = {
  transport: 'local',
  layers: [],
  terminalFamily: 'unknown',
  keyboard: {
    encoding: 'legacy',
    pasteShortcutShapes: ['ctrl+shift+v', 'alt+v']
  },
  paste: {
    bracketedPaste: true
  },
  copy: {
    writePath: 'native',
    readPath: 'native',
    copyOnSelect: false
  },
  mouse: {
    tracking: false,
    shiftDragHint: false
  },
  diagnostics: []
}

const makeCaps = (overrides: Partial<TerminalCapabilities> = {}): TerminalCapabilities => ({
  ...baseCaps,
  ...overrides,
  keyboard: {
    ...baseCaps.keyboard,
    ...overrides.keyboard
  },
  paste: {
    ...baseCaps.paste,
    ...overrides.paste
  },
  copy: {
    ...baseCaps.copy,
    ...overrides.copy
  },
  mouse: {
    ...baseCaps.mouse,
    ...overrides.mouse
  },
  diagnostics: overrides.diagnostics ?? baseCaps.diagnostics
})

describe('classifyKeyEvent', () => {
  it('classifies bracketed paste before any shortcut modifiers', () => {
    const result = classifyKeyEvent({
      input: 'c',
      raw: '\x1b[200~pasted text',
      key: { ctrl: true, shift: false, meta: false, super: false },
      caps: baseCaps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'paste', source: 'bracketed' })
  })

  it('classifies Cmd+V on macOS as hotkey paste', () => {
    const caps = makeCaps({
      terminalFamily: 'iterm2',
      keyboard: {
        encoding: 'csi-u',
        pasteShortcutShapes: ['cmd+v', 'ctrl+shift+v']
      }
    })

    const result = classifyKeyEvent({
      input: 'v',
      raw: 'v',
      key: { ctrl: false, shift: false, meta: true, super: false },
      caps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'paste', source: 'hotkey' })
  })

  it('classifies Ctrl+Shift+V on Linux as hotkey paste', () => {
    const result = classifyKeyEvent({
      input: 'v',
      raw: 'v',
      key: { ctrl: true, shift: true, meta: false, super: false },
      caps: baseCaps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'paste', source: 'hotkey' })
  })

  it('classifies kitty protocol Ctrl+V as hotkey paste when the capability advertises kitty keyboard support', () => {
    const caps = makeCaps({
      terminalFamily: 'kitty',
      keyboard: {
        encoding: 'kitty',
        pasteShortcutShapes: ['ctrl+shift+v', 'alt+v', 'ctrl+v']
      }
    })

    const result = classifyKeyEvent({
      input: 'v',
      raw: '\x1b[118;5u',
      key: { ctrl: true, shift: false, meta: false, super: false },
      caps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'paste', source: 'hotkey' })
  })

  it('classifies Cmd+C with a selection as copy', () => {
    const caps = makeCaps({
      terminalFamily: 'iterm2',
      keyboard: {
        encoding: 'csi-u',
        pasteShortcutShapes: ['cmd+v', 'ctrl+shift+v']
      }
    })

    const result = classifyKeyEvent({
      input: 'c',
      raw: '\x1b[99;9u',
      key: { ctrl: false, shift: false, meta: true, super: false },
      caps,
      state: { hasSelection: true, busy: false }
    })

    expect(result).toEqual({ type: 'copy' })
  })

  it('classifies Cmd+C without a selection as noop', () => {
    const caps = makeCaps({
      terminalFamily: 'iterm2',
      keyboard: {
        encoding: 'csi-u',
        pasteShortcutShapes: ['cmd+v', 'ctrl+shift+v']
      }
    })

    const result = classifyKeyEvent({
      input: 'c',
      raw: '\x1b[99;9u',
      key: { ctrl: false, shift: false, meta: false, super: true },
      caps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'noop' })
  })

  it('classifies Ctrl+Shift+C on non-mac as copy when a selection exists', () => {
    const result = classifyKeyEvent({
      input: 'c',
      raw: '\x1b[99;6u',
      key: { ctrl: true, shift: true, meta: false, super: false },
      caps: baseCaps,
      state: { hasSelection: true, busy: false }
    })

    expect(result).toEqual({ type: 'copy' })
  })

  it('keeps forwarded Cmd+C over SSH from becoming interrupt', () => {
    const remoteCaps = makeCaps({
      transport: 'ssh',
      layers: ['ssh'],
      terminalFamily: 'iterm2',
      keyboard: {
        encoding: 'csi-u',
        pasteShortcutShapes: ['cmd+v', 'ctrl+shift+v']
      }
    })

    const result = classifyKeyEvent({
      input: 'c',
      raw: '\x1b[99;9u',
      key: { ctrl: true, shift: false, meta: false, super: true },
      caps: remoteCaps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'noop' })
  })

  it('classifies plain Ctrl+C as interrupt', () => {
    const result = classifyKeyEvent({
      input: 'c',
      raw: '\x03',
      key: { ctrl: true, shift: false, meta: false, super: false },
      caps: baseCaps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'interrupt' })
  })

  it('returns text for ordinary input', () => {
    const result = classifyKeyEvent({
      input: 'hello',
      raw: 'hello',
      key: { ctrl: false, shift: false, meta: false, super: false },
      caps: baseCaps,
      state: { hasSelection: false, busy: false }
    })

    expect(result).toEqual({ type: 'text', text: 'hello' })
  })
})
