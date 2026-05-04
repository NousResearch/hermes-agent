import { describe, expect, it } from 'vitest'

import { collectTerminalSignals } from '../lib/terminalSignals.js'

describe('collectTerminalSignals', () => {
  it('collects local terminal facts without SSH or multiplexer env', () => {
    const signals = collectTerminalSignals({
      env: {
        TERM: 'xterm-256color',
        TERM_PROGRAM: 'iTerm.app',
        TERM_PROGRAM_VERSION: '3.5.8'
      },
      platform: 'darwin',
      isStdinTty: true,
      isStdoutTty: true,
      shellExecutable: '/bin/zsh',
      shellArgv0: '-zsh'
    })

    expect(signals.platform).toBe('darwin')
    expect(signals.isStdinTty).toBe(true)
    expect(signals.isStdoutTty).toBe(true)
    expect(signals.ssh).toEqual({
      hasSshConnection: false,
      hasSshClient: false,
      hasSshTty: false
    })
    expect(signals.multiplexer).toEqual({ tmux: false, screen: false, zellij: false, cy: false })
    expect(signals.env).toEqual({
      TERM: 'xterm-256color',
      TERM_PROGRAM: 'iTerm.app',
      TERM_PROGRAM_VERSION: '3.5.8'
    })
    expect(signals.shell.family).toBe('zsh')
    expect(signals.shell.login).toBe(true)
  })

  it('captures SSH connection details explicitly', () => {
    const signals = collectTerminalSignals({
      env: {
        SSH_CONNECTION: '1.1.1.1 555 2.2.2.2 22',
        SSH_CLIENT: '1.1.1.1 555 22',
        SSH_TTY: '/dev/pts/4',
        TERM: 'xterm-256color'
      },
      platform: 'linux'
    })

    expect(signals.ssh).toEqual({
      hasSshConnection: true,
      hasSshClient: true,
      hasSshTty: true,
      tty: '/dev/pts/4',
      connection: '1.1.1.1 555 2.2.2.2 22'
    })
    expect(signals.env.TERM).toBe('xterm-256color')
  })

  it('detects tmux and screen as nested transport hints', () => {
    expect(collectTerminalSignals({ env: { TMUX: '/tmp/tmux,1,0' }, platform: 'linux' }).multiplexer.tmux).toBe(true)
    expect(collectTerminalSignals({ env: { STY: '123.pts-0.host' }, platform: 'linux' }).multiplexer.screen).toBe(true)
  })

  it('collects all supported terminal env hints when present', () => {
    const env = {
      TERM: 'xterm-kitty',
      COLORTERM: 'truecolor',
      TERM_PROGRAM: 'ghostty',
      TERM_PROGRAM_VERSION: '1.2.3',
      VTE_VERSION: '7600',
      WT_SESSION: 'wt-session',
      KITTY_WINDOW_ID: '42',
      WEZTERM_PANE: 'wezterm-pane',
      GHOSTTY_RESOURCES_DIR: '/Applications/Ghostty.app',
      KONSOLE_VERSION: '24.12',
      ITERM_SESSION_ID: 'iterm-session',
      LC_TERMINAL: 'iTerm2',
      TERM_SESSION_ID: 'session-123'
    }

    const signals = collectTerminalSignals({ env, platform: 'linux' })

    expect(signals.env).toEqual(env)
  })

  it('returns empty signal sets for an empty env snapshot', () => {
    const signals = collectTerminalSignals({ env: {}, platform: 'linux' })

    expect(signals.platform).toBe('linux')
    expect(signals.isStdinTty).toBe(false)
    expect(signals.isStdoutTty).toBe(false)
    expect(signals.ssh).toEqual({
      hasSshConnection: false,
      hasSshClient: false,
      hasSshTty: false
    })
    expect(signals.multiplexer).toEqual({ tmux: false, screen: false, zellij: false, cy: false })
    expect(signals.env).toEqual({})
    expect(signals.shell.family).toBe('unknown')
    expect(signals.shell.startupRiskHints).toEqual([])
  })
})
