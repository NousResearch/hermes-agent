import { describe, expect, it } from 'vitest'

import { shouldUseAnsiDim } from './Text.js'

describe('shouldUseAnsiDim', () => {
  it('disables ANSI dim on VTE terminals by default', () => {
    expect(shouldUseAnsiDim({ VTE_VERSION: '7603' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('keeps ANSI dim enabled elsewhere by default', () => {
    expect(shouldUseAnsiDim({ TERM: 'xterm-256color' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('honors explicit env override', () => {
    expect(shouldUseAnsiDim({ HERMES_TUI_DIM: '1', VTE_VERSION: '7603' } as NodeJS.ProcessEnv)).toBe(true)
    expect(shouldUseAnsiDim({ HERMES_TUI_DIM: '0' } as NodeJS.ProcessEnv)).toBe(false)
  })
})
