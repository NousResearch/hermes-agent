import { describe, expect, it } from 'vitest'

import { isTermuxEnv, isTermuxTuiMode } from '../lib/termux.js'

describe('isTermuxEnv', () => {
  it('detects TERMUX_VERSION marker', () => {
    expect(isTermuxEnv({ TERMUX_VERSION: '0.118.0' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('detects Termux PREFIX path marker', () => {
    expect(isTermuxEnv({ PREFIX: '/data/data/com.termux/files/usr' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('returns false for generic Linux envs', () => {
    expect(isTermuxEnv({ PREFIX: '/usr' } as NodeJS.ProcessEnv)).toBe(false)
  })
})

describe('isTermuxTuiMode', () => {
  it('defaults to true inside Termux', () => {
    expect(isTermuxTuiMode({ TERMUX_VERSION: '0.118.0' } as NodeJS.ProcessEnv)).toBe(true)
  })

  it('allows explicit opt-out override', () => {
    expect(isTermuxTuiMode({ TERMUX_VERSION: '0.118.0', HERMES_TUI_TERMUX_MODE: '0' } as NodeJS.ProcessEnv)).toBe(false)
  })

  it('stays false outside Termux even if override is set', () => {
    expect(isTermuxTuiMode({ HERMES_TUI_TERMUX_MODE: '1', PREFIX: '/usr' } as NodeJS.ProcessEnv)).toBe(false)
  })
})
