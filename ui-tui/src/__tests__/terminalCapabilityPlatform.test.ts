import { beforeEach, describe, expect, it } from 'vitest'

import { coreCommands } from '../app/slash/commands/core.js'
import { $terminalEnvironment, createTerminalEnvironment, updateTerminalProbe } from '../app/terminalEnvironmentStore.js'
import { buildHelpHintHotkeys } from '../content/hotkeys.js'
import { deriveTerminalCapabilities } from '../lib/terminalCapabilities.js'
import { formatTerminalDoctor } from '../lib/terminalDoctor.js'
import { probeTerminalCapabilities, type Querier } from '../lib/terminalProbe.js'
import { collectTerminalSignals } from '../lib/terminalSignals.js'

const makeSignals = (env: NodeJS.ProcessEnv, platform: NodeJS.Platform = 'linux') =>
  collectTerminalSignals({ env, platform, isStdinTty: true, isStdoutTty: true })

const makeSlashCtx = () => {
  const panels: Array<{ header: string; sections: Array<{ rows?: [string, string][]; text?: string; title?: string }> }> = []
  const pages: Array<{ output: string; title: string }> = []

  return {
    ctx: {
      local: { catalog: { categories: [], skillCount: 0 } },
      transcript: {
        page: (output: string, title: string) => pages.push({ output, title }),
        panel: (header: string, sections: Array<{ rows?: [string, string][]; text?: string; title?: string }>) =>
          panels.push({ header, sections }),
        sys: () => {}
      },
      ui: { theme: { brand: { helpHeader: 'Help' } } }
    } as any,
    pages,
    panels
  }
}

describe('terminal capability platform', () => {
  beforeEach(() => {
    $terminalEnvironment.set(createTerminalEnvironment({ TERM: 'xterm-256color' }))
  })

  it('derives ordered layers, terminal family, copy paths, and dynamic hotkey hints', () => {
    const signals = makeSignals({
      LC_TERMINAL: 'iTerm2',
      SHELL: '/bin/zsh',
      SSH_CONNECTION: '2001:db8::1 51000 10.0.0.5 22',
      SSH_TTY: '/dev/pts/3',
      TERM: 'screen-256color',
      TMUX: '/tmp/tmux-1000/default,123,0',
      ZSH_VERSION: '5.9'
    })

    const capabilities = deriveTerminalCapabilities(signals, { osc52ReadSupported: true })

    expect(capabilities.layers).toEqual(['ssh', 'tmux'])
    expect(capabilities.transport).toBe('tmux')
    expect(capabilities.terminalFamily).toBe('tmux')
    expect(capabilities.copy.writePath).toBe('tmux-buffer')
    expect(capabilities.copy.readPath).toBe('osc52-query')

    const hotkeys = buildHelpHintHotkeys({ signals, capabilities })
    expect(hotkeys).toContainEqual(['Ctrl+Shift+V', 'paste text; /paste attaches clipboard image'])
    expect(hotkeys).toContainEqual(['tmux', 'copy uses tmux-buffer (write: tmux-buffer)'])
  })

  it('formats /doctor output without leaking IPv4 or IPv6 SSH endpoints', () => {
    const env = createTerminalEnvironment({
      LC_TERMINAL: 'iTerm2',
      SHELL: '/bin/zsh',
      SSH_CONNECTION: '2001:db8::1 51000 10.0.0.5 22',
      SSH_TTY: '/dev/pts/3',
      TERM: 'screen-256color',
      TMUX: '/tmp/tmux-1000/default,123,0',
      ZSH_VERSION: '5.9'
    })

    const output = formatTerminalDoctor(env)

    expect(output).toContain('=== Terminal Doctor ===')
    expect(output).toContain('layers: ssh, tmux')
    expect(output).toContain('shell: zsh 5.9')
    expect(output).not.toContain('2001:db8::1')
    expect(output).not.toContain('10.0.0.5')
    expect(output).toContain('SSH_CONNECTION: <redacted> 51000 <redacted> 22')
  })

  it('probes bracketed paste, kitty keyboard flags, xtversion, and OSC52 read support', async () => {
    const querier: Querier = {
      flush: async () => {},
      send: async query => {
        if (query.request.includes('>0q')) {return '\u001bP>|kitty(0.34.1)\u001b\\' as never}

        if (query.request.includes('?2004$p')) {return '\u001b[?2004;1$y' as never}

        if (query.request.includes('?u')) {return '\u001b[?7u' as never}

        if (query.request.includes(']52;c;?')) {return '\u001b]52;c;Zm9v\u0007' as never}

        return undefined
      }
    }

    const result = await probeTerminalCapabilities(querier, { allowOsc52Read: true, timeoutMs: 25 })

    expect(result.xtversionName).toBe('kitty(0.34.1)')
    expect(result.bracketedPaste).toBe(true)
    expect(result.kittyKeyboardFlags).toBe(7)
    expect(result.osc52ReadSupported).toBe(true)
  })

  it('updates the terminal environment store after a gated probe result', () => {
    $terminalEnvironment.set(
      createTerminalEnvironment({ SSH_CONNECTION: '127.0.0.1 1 127.0.0.1 22', SSH_TTY: '/dev/pts/1', TERM: 'xterm-256color' })
    )

    updateTerminalProbe({ osc52ReadSupported: true })

    expect($terminalEnvironment.get().probe.osc52ReadSupported).toBe(true)
    expect($terminalEnvironment.get().capabilities.copy.readPath).toBe('osc52-query')
  })

  it('wires /doctor and capability-aware hotkeys into local slash commands', () => {
    const { ctx, pages, panels } = makeSlashCtx()

    coreCommands.find(command => command.name === 'doctor')?.run('', ctx, '/doctor')
    expect(pages[0]).toMatchObject({ title: 'Terminal Doctor' })
    expect(pages[0]?.output).toContain('=== Terminal Doctor ===')

    coreCommands.find(command => command.name === 'help')?.run('', ctx, '/help')
    const hotkeyRows = panels[0]?.sections.find(section => section.title === 'Hotkeys')?.rows ?? []

    expect(hotkeyRows).toContainEqual(['Ctrl+Shift+V', 'paste text; /paste attaches clipboard image'])
  })
})
