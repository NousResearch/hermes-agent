import { afterEach, describe, expect, it } from 'vitest'

import { reorderBidi, resetBidiDetectionForTest, shouldReorderBidi } from './bidi.js'

type ClusteredChar = { value: string; width: number; styleId: number; hyperlink: string | undefined }

const toChars = (value: string): ClusteredChar[] =>
  [...value].map(ch => ({ value: ch, width: 1, styleId: 0, hyperlink: undefined }))

const reorderAs = (input: string, env: NodeJS.ProcessEnv): string => {
  process.env = { ...env } as NodeJS.ProcessEnv
  resetBidiDetectionForTest()

  return reorderBidi(toChars(input))
    .map(c => c.value)
    .join('')
}

afterEach(() => {
  resetBidiDetectionForTest()
})

describe('shouldReorderBidi', () => {
  it('reorders on terminals known to lack bidi (regression: gate was Windows-only)', () => {
    // Plain Linux terminals and Windows-family terminals all reach the screen
    // in logical order, so software reordering must run here.
    expect(shouldReorderBidi({ TERM: 'xterm' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ TERM: 'xterm-kitty' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ TERM: 'alacritty' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ TERM_PROGRAM: 'ghostty' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ WT_SESSION: '1' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('still reorders where bidi exists but is off by default', () => {
    // WezTerm implements bidi but ships it disabled (bidi_enabled = false).
    expect(shouldReorderBidi({ TERM_PROGRAM: 'WezTerm' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('leaves terminals that do bidi themselves untouched', () => {
    expect(shouldReorderBidi({ TERM_PROGRAM: 'iTerm.app' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldReorderBidi({ TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe(false)
    // VTE-based terminals (GNOME Terminal, Tilix, Xfce Terminal, Guake…) since VTE 0.58.
    expect(shouldReorderBidi({ VTE_VERSION: '7603' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldReorderBidi({ KONSOLE_VERSION: '221201' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldReorderBidi({ TERM: 'mlterm' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('honors the HERMES_TUI_BIDI override', () => {
    expect(shouldReorderBidi({ HERMES_TUI_BIDI: '1', TERM_PROGRAM: 'iTerm.app' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ HERMES_TUI_BIDI: '0', TERM: 'xterm-kitty' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldReorderBidi({ HERMES_TUI_BIDI: 'on' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ HERMES_TUI_BIDI: 'off', TERM: 'xterm' } as NodeJS.ProcessEnv)).toBe(false)
  })
})

describe('reorderBidi', () => {
  it('reverses a pure Hebrew run so it renders right-to-left', () => {
    // Logical: shin lamed vav mem. Visual for an LTR-only terminal: reversed.
    expect(reorderAs('שלום', { TERM: 'xterm' } as NodeJS.ProcessEnv)).toBe('םולש')
  })

  it('keeps LTR prefixes in place next to a Hebrew run', () => {
    expect(reorderAs('abc שלום def', { TERM: 'xterm' } as NodeJS.ProcessEnv)).toBe('abc םולש def')
  })

  it('leaves pure-LTR text untouched', () => {
    expect(reorderAs('hello world', { TERM: 'xterm' } as NodeJS.ProcessEnv)).toBe('hello world')
  })

  it('is a no-op when reordering is disabled', () => {
    expect(reorderAs('שלום', { HERMES_TUI_BIDI: '0' } as NodeJS.ProcessEnv)).toBe('שלום')
  })

  it('is a no-op on terminals with native bidi', () => {
    expect(reorderAs('שלום', { TERM_PROGRAM: 'iTerm.app' } as NodeJS.ProcessEnv)).toBe('שלום')
  })
})
