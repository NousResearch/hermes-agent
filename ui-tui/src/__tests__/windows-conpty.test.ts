import { describe, expect, it } from 'vitest'

import { isWindowsConptyTuiMode } from '../lib/windows-conpty.js'

describe('isWindowsConptyTuiMode', () => {
  it('is false on non-Windows hosts', () => {
    expect(isWindowsConptyTuiMode({} as NodeJS.ProcessEnv, 'linux')).toBe(false)
    expect(isWindowsConptyTuiMode({ WT_SESSION: '1' } as NodeJS.ProcessEnv, 'darwin')).toBe(false)
  })

  it('defaults to true on native Windows ConPTY (cmd/powershell/conhost)', () => {
    expect(isWindowsConptyTuiMode({} as NodeJS.ProcessEnv, 'win32')).toBe(true)
  })

  it('is true inside Windows Terminal even with MSYSTEM set', () => {
    expect(isWindowsConptyTuiMode({ MSYSTEM: 'MINGW64', WT_SESSION: 'abc' } as NodeJS.ProcessEnv, 'win32')).toBe(true)
  })

  it('is false for Git Bash/MSYS mintty outside Windows Terminal', () => {
    expect(isWindowsConptyTuiMode({ MSYSTEM: 'MINGW64' } as NodeJS.ProcessEnv, 'win32')).toBe(false)
  })
})
