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
    // iTerm2's bidi is opt-in (Settings > Advanced > Support right-to-left-text)
    // and doesn't cover the alternate screen, which fullscreen mode uses.
    expect(shouldReorderBidi({ TERM_PROGRAM: 'iTerm.app' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('leaves terminals that do bidi themselves untouched', () => {
    expect(shouldReorderBidi({ TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe(false)
    // VTE-based terminals (GNOME Terminal, Tilix, Xfce Terminal, Guake…) since VTE 0.58.
    expect(shouldReorderBidi({ VTE_VERSION: '7603' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldReorderBidi({ VTE_VERSION: '5800' } as NodeJS.ProcessEnv)).toBe(false)
    // Konsole reorders through ICU from 22.12.
    expect(shouldReorderBidi({ KONSOLE_VERSION: '221200' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldReorderBidi({ KONSOLE_VERSION: '240800' } as NodeJS.ProcessEnv)).toBe(false)
    expect(shouldReorderBidi({ TERM: 'mlterm' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('reorders on builds older than their terminal bidi support', () => {
    // VTE 0.50 has no bidi, and fribidi stays optional past 0.58 — a build
    // without it still reports a version but emits logical order.
    expect(shouldReorderBidi({ VTE_VERSION: '5003' } as NodeJS.ProcessEnv)).toBe(true)
    // Konsole before 22.12 reordered per run of identical attributes, so a
    // coloured line came out wrong (KDE bug 403729).
    expect(shouldReorderBidi({ KONSOLE_VERSION: '211202' } as NodeJS.ProcessEnv)).toBe(true)
    // Unparseable version = unknown, and unknown means we reorder.
    expect(shouldReorderBidi({ VTE_VERSION: 'nonsense' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldReorderBidi({ KONSOLE_VERSION: '' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('honors the HERMES_TUI_BIDI override', () => {
    expect(shouldReorderBidi({ HERMES_TUI_BIDI: '1', TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe(true)
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
    expect(reorderAs('שלום', { TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe('שלום')
  })
})

describe('reorderBidi mirroring (UAX #9 rule L4)', () => {
  // Reordering alone is not enough: a bracket pair that resolves into an RTL
  // run must also be glyph-mirrored, or it reads swapped — ")םלוע( םולש"
  // instead of "(םלוע) םולש". Expected values are the visual strings produced
  // by GNU FriBidi (`fribidi --ltr --nopad --nobreak`), and every case here is
  // base-direction independent, so it holds under the UAX#9 first-strong rule
  // as-is (the base policy is tracked separately, see #72508).
  const env = { TERM: 'xterm' } as NodeJS.ProcessEnv

  it('mirrors a bracket pair that resolves into the RTL run', () => {
    expect(reorderAs('שלום (עולם)', env)).toBe('(םלוע) םולש')
    expect(reorderAs('│ שלום (עולם) │', env)).toBe('│ (םלוע) םולש │')
    expect(reorderAs('א (ב) ג (ד) ה', env)).toBe('ה (ד) ג (ב) א')
    expect(reorderAs('שלום (עולם), עולם (שלום)', env)).toBe('(םולש) םלוע ,(םלוע) םולש')
    expect(reorderAs('שלום (עולם (עולם))', env)).toBe('((םלוע) םלוע) םולש')
  })

  it('leaves a bracket pair that resolves to the paragraph direction', () => {
    // Brackets preceded only by the paragraph start stay at level 0.
    expect(reorderAs('(שלום)', env)).toBe('(םולש)')
    expect(reorderAs('עולם (שלום) עולם', env)).toBe('םלוע (םולש) םלוע')
    expect(reorderAs('שלום [עולם] עולם', env)).toBe('םלוע [םלוע] םולש')
  })

  it('does not mirror when reordering is skipped or disabled', () => {
    expect(reorderAs('שלום (עולם)', { HERMES_TUI_BIDI: '0' } as NodeJS.ProcessEnv)).toBe('שלום (עולם)')
    expect(reorderAs('hello (world)', env)).toBe('hello (world)')
  })
})
