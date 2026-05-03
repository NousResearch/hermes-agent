import { describe, expect, it } from 'vitest'

import { detectShellSignals } from '../lib/shellSignals.js'

describe('detectShellSignals', () => {
  it('detects bash from BASH_VERSION and reports its version', () => {
    const result = detectShellSignals({
      env: { BASH_VERSION: '5.2.0', SHELL: '/bin/zsh' },
      executable: '/bin/sh',
      argv0: '-bash'
    })

    expect(result.family).toBe('bash')
    expect(result.version).toBe('5.2.0')
    expect(result.login).toBe(true)
  })

  it('detects zsh from ZSH_VERSION and reports its version', () => {
    const result = detectShellSignals({
      env: { ZSH_VERSION: '5.9', SHELL: '/bin/bash' },
      executable: '/bin/sh',
      argv0: 'zsh'
    })

    expect(result.family).toBe('zsh')
    expect(result.version).toBe('5.9')
    expect(result.login).toBe(false)
  })

  it('detects fish from the executable name', () => {
    const result = detectShellSignals({
      env: { FISH_VERSION: '3.7.1' },
      executable: '/usr/local/bin/fish',
      argv0: 'fish'
    })

    expect(result.family).toBe('fish')
    expect(result.version).toBe('3.7.1')
    expect(result.executable).toBe('/usr/local/bin/fish')
  })

  it('detects posix-sh shells from dash and ash executable names', () => {
    expect(detectShellSignals({ env: {}, executable: '/bin/dash', argv0: 'sh' }).family).toBe('posix-sh')
    expect(detectShellSignals({ env: {}, executable: '/bin/ash', argv0: 'ash' }).family).toBe('posix-sh')
  })

  it('detects powershell from the executable name', () => {
    const result = detectShellSignals({
      env: {},
      executable: 'C:\\Program Files\\PowerShell\\7\\pwsh.exe',
      argv0: 'pwsh'
    })

    expect(result.family).toBe('powershell')
    expect(result.executable).toBe('C:\\Program Files\\PowerShell\\7\\pwsh.exe')
  })

  it('detects tcsh from the executable name', () => {
    const result = detectShellSignals({
      env: {},
      executable: '/bin/tcsh',
      argv0: 'tcsh'
    })

    expect(result.family).toBe('tcsh')
  })

  it('falls back to unknown when nothing matches', () => {
    const result = detectShellSignals({ env: {}, executable: '/usr/bin/node', argv0: 'node' })

    expect(result.family).toBe('unknown')
    expect(result.startupRiskHints).toEqual([])
  })

  it('marks login shells when argv0 starts with a dash', () => {
    const result = detectShellSignals({ env: {}, executable: '/bin/bash', argv0: '-bash' })

    expect(result.login).toBe(true)
  })

  it('uses $SHELL as a fallback executable hint without treating it as proof', () => {
    const result = detectShellSignals({ env: { SHELL: '/bin/zsh' } })

    expect(result.executable).toBe('/bin/zsh')
    expect(result.family).toBe('unknown')
  })

  it.each([
    [
      'bash',
      { env: { BASH_VERSION: '5.2.0' }, executable: '/bin/sh' },
      '.bashrc can reset stty or TERM'
    ],
    [
      'zsh',
      { env: { ZSH_VERSION: '5.9' }, executable: '/bin/sh' },
      '.zshenv always runs; check for output or stty mutations'
    ],
    [
      'fish',
      { env: {}, executable: '/usr/bin/fish' },
      'config.fish can run broadly; unguarded output breaks noninteractive commands'
    ],
    [
      'posix-sh',
      { env: {}, executable: '/bin/dash' },
      'minimal editor support; no reliable shell-layer bracketed paste'
    ],
    [
      'powershell',
      { env: {}, executable: 'C:\\Program Files\\PowerShell\\7\\pwsh.exe' },
      "PSReadLine doesn't apply when Hermes owns raw mode"
    ],
    [
      'tcsh',
      { env: {}, executable: '/bin/tcsh' },
      'legacy line editing; startup files often contain old stty/tset calls'
    ]
  ] as const)('populates startup risk hints for %s', (_family, input, expectedHint) => {
    const result = detectShellSignals(input)

    expect(result.startupRiskHints).toEqual([expectedHint])
  })
})
