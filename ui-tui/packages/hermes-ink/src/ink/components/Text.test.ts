import { afterEach, describe, expect, it } from 'vitest'

import { dimColorFallback, setDimFallbackColor, shouldUseAnsiDim } from './Text.js'

describe('shouldUseAnsiDim', () => {
  it('disables ANSI dim on VTE terminals by default', () => {
    expect(shouldUseAnsiDim({ VTE_VERSION: '7603' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('disables ANSI dim on Apple Terminal by default', () => {
    expect(shouldUseAnsiDim({ TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('keeps ANSI dim enabled elsewhere by default', () => {
    expect(shouldUseAnsiDim({ TERM: 'xterm-256color' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('honors explicit env override', () => {
    expect(shouldUseAnsiDim({ HERMES_TUI_DIM: '1', VTE_VERSION: '7603' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldUseAnsiDim({ HERMES_TUI_DIM: '1', TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldUseAnsiDim({ HERMES_TUI_DIM: '0' } as NodeJS.ProcessEnv)).toBe(false)
  })
})

describe('dimColorFallback', () => {
  afterEach(() => {
    setDimFallbackColor(undefined)
  })

  it('renders Apple Terminal dim as muted gray by default', () => {
    expect(dimColorFallback({ TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe('#6B7280')
  })

  it('normalizes Apple Terminal names before matching', () => {
    expect(dimColorFallback({ TERM_PROGRAM: ' Apple_Terminal ' } as NodeJS.ProcessEnv)).toBe('#6B7280')
  })

  it('does not apply when dim is explicitly configured', () => {
    expect(
      dimColorFallback({ HERMES_TUI_DIM: '1', TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)
    ).toBeUndefined()
    expect(
      dimColorFallback({ HERMES_TUI_DIM: '0', TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)
    ).toBeUndefined()
  })

  it('uses the theme tone once one is supplied, so dim stays in-palette', () => {
    setDimFallbackColor('#936e06')

    expect(dimColorFallback({ TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe('#936e06')
  })

  it('falls back to the boot default when the theme tone is cleared', () => {
    setDimFallbackColor('#936e06')
    setDimFallbackColor(undefined)

    expect(dimColorFallback({ TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv)).toBe('#6B7280')
  })

  it('stays inert on terminals that honor SGR 2, whatever the theme tone', () => {
    setDimFallbackColor('#936e06')

    expect(dimColorFallback({ TERM: 'xterm-256color' } as NodeJS.ProcessEnv)).toBeUndefined()
  })
})
