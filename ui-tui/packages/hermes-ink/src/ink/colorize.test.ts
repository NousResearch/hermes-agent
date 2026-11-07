import { describe, expect, it } from 'vitest'

import {
  richEightBitColorNumber,
  shouldClampChalkLevelForTmux,
  shouldUseRichEightBitDowngradeForLegacyAppleTerminal
} from './colorize.js'

describe('shouldClampChalkLevelForTmux', () => {
  it('keeps truecolor when explicitly enabled inside tmux', () => {
    expect(shouldClampChalkLevelForTmux({ HERMES_TUI_TRUECOLOR: '1', TMUX: '/tmp/tmux-1000/default,1,0' }, 3)).toBe(false)
    expect(shouldClampChalkLevelForTmux({ HERMES_TUI_TRUECOLOR: ' true ', TMUX: 'tmux' }, 3)).toBe(false)
  })

  it('keeps the safe 256-color default inside tmux without the opt-in', () => {
    expect(shouldClampChalkLevelForTmux({ TMUX: '/tmp/tmux-1000/default,1,0' }, 3)).toBe(true)
    expect(shouldClampChalkLevelForTmux({ HERMES_TUI_TRUECOLOR: '0', TMUX: 'tmux' }, 3)).toBe(true)
    expect(shouldClampChalkLevelForTmux({ HERMES_TUI_TRUECOLOR: '1', NO_COLOR: '1', TMUX: 'tmux' }, 3)).toBe(true)
  })
})

describe('shouldUseRichEightBitDowngradeForLegacyAppleTerminal', () => {
  it('uses Rich-compatible 256-color downgrade on legacy Apple Terminal', () => {
    expect(
      shouldUseRichEightBitDowngradeForLegacyAppleTerminal({ TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv, 2)
    ).toBe(true)
  })

  it('normalizes Apple Terminal names before matching', () => {
    expect(
      shouldUseRichEightBitDowngradeForLegacyAppleTerminal({ TERM_PROGRAM: ' Apple_Terminal ' } as NodeJS.ProcessEnv, 2)
    ).toBe(true)
  })

  it('does not rewrite when Apple Terminal advertises truecolor', () => {
    expect(
      shouldUseRichEightBitDowngradeForLegacyAppleTerminal(
        { COLORTERM: 'truecolor', TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv,
        3
      )
    ).toBe(false)
  })

  it('does not override explicit color environment choices', () => {
    expect(
      shouldUseRichEightBitDowngradeForLegacyAppleTerminal(
        { FORCE_COLOR: '2', TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv,
        2
      )
    ).toBe(false)
    expect(
      shouldUseRichEightBitDowngradeForLegacyAppleTerminal(
        { HERMES_TUI_TRUECOLOR: '1', TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv,
        3
      )
    ).toBe(false)
  })
})

describe('richEightBitColorNumber', () => {
  it('matches Rich downgrade output for default Hermes skin colors', () => {
    expect(richEightBitColorNumber(0xff, 0xd7, 0x00)).toBe(220)
    expect(richEightBitColorNumber(0xff, 0xbf, 0x00)).toBe(214)
    expect(richEightBitColorNumber(0xcd, 0x7f, 0x32)).toBe(173)
    expect(richEightBitColorNumber(0xb8, 0x86, 0x0b)).toBe(136)
    expect(richEightBitColorNumber(0xff, 0xf8, 0xdc)).toBe(230)
  })
})
