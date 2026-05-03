import { describe, expect, it } from 'vitest'

import { collectTerminalSignals } from '../lib/terminalSignals.js'
import { deriveTerminalCapabilities } from '../lib/terminalCapabilities.js'
import { formatTerminalDoctor } from '../lib/terminalDoctor.js'

describe('formatTerminalDoctor', () => {
  it('redacts raw SSH connection while showing ssh transport', () => {
    const signals = collectTerminalSignals({
      env: { SSH_CONNECTION: '1.1.1.1 555 2.2.2.2 22', SSH_TTY: '/dev/pts/4', TERM: 'xterm-256color' },
      platform: 'linux'
    })
    const caps = deriveTerminalCapabilities(signals)

    const text = formatTerminalDoctor({ signals, probe: {}, capabilities: caps })

    expect(text).toContain('transport: ssh')
    expect(text).not.toContain('1.1.1.1')
  })

  it('includes terminal family and shell info', () => {
    const signals = collectTerminalSignals({
      env: { TERM_PROGRAM: 'iTerm.app', SHELL: '/bin/zsh', ZSH_VERSION: '5.9' },
      platform: 'darwin',
      shellExecutable: '/bin/zsh',
      shellArgv0: '-zsh'
    })
    const caps = deriveTerminalCapabilities(signals)

    const text = formatTerminalDoctor({ signals, probe: {}, capabilities: caps })

    expect(text).toContain('terminal: iterm2')
    expect(text).toContain('shell: zsh')
    expect(text).not.toContain('SSH_CONNECTION')
  })

  it('shows clipboard routes', () => {
    const signals = collectTerminalSignals({
      env: { TMUX: '/tmp/tmux,1,0', TERM: 'screen-256color' },
      platform: 'linux'
    })
    const caps = deriveTerminalCapabilities(signals)

    const text = formatTerminalDoctor({ signals, probe: {}, capabilities: caps })

    expect(text).toContain('clipboard write: tmux-buffer')
    expect(text).toContain('keyboard: legacy')
  })
})
