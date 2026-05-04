import { describe, expect, it } from 'vitest'

import type { TerminalProbeResult } from '../lib/terminalCapabilities.js'
import { deriveTerminalCapabilities } from '../lib/terminalCapabilities.js'
import type { TerminalSignals } from '../lib/terminalSignals.js'

type SignalOverrides = {
  platform?: NodeJS.Platform
  isStdinTty?: boolean
  isStdoutTty?: boolean
  ssh?: Partial<TerminalSignals['ssh']>
  multiplexer?: Partial<TerminalSignals['multiplexer']>
  env?: Partial<TerminalSignals['env']>
  shell?: Partial<TerminalSignals['shell']>
}

const makeSignals = (overrides: SignalOverrides = {}): TerminalSignals => ({
  platform: overrides.platform ?? 'linux',
  isStdinTty: overrides.isStdinTty ?? true,
  isStdoutTty: overrides.isStdoutTty ?? true,
  ssh: {
    hasSshConnection: false,
    hasSshClient: false,
    hasSshTty: false,
    ...((overrides.ssh ?? {}) as TerminalSignals['ssh'])
  },
  multiplexer: {
    tmux: false,
    screen: false,
    zellij: false,
    cy: false,
    ...((overrides.multiplexer ?? {}) as TerminalSignals['multiplexer'])
  },
  env: {
    ...((overrides.env ?? {}) as TerminalSignals['env'])
  },
  shell: {
    family: 'unknown',
    startupRiskHints: [],
    ...((overrides.shell ?? {}) as TerminalSignals['shell'])
  }
})

describe('deriveTerminalCapabilities', () => {
  it('detects local macOS iTerm2 with CSI-u keyboard encoding and native clipboard', () => {
    const caps = deriveTerminalCapabilities(
      makeSignals({
        platform: 'darwin',
        env: {
          TERM: 'xterm-256color',
          TERM_PROGRAM: 'iTerm.app',
          ITERM_SESSION_ID: 'abc123',
          TERM_PROGRAM_VERSION: '3.5.1'
        }
      })
    )

    expect(caps).toMatchObject({
      transport: 'local',
      terminalFamily: 'iterm2',
      terminalVersion: '3.5.1',
      keyboard: {
        encoding: 'csi-u'
      },
      paste: {
        bracketedPaste: true
      },
      copy: {
        writePath: 'native',
        readPath: 'native',
        copyOnSelect: true
      },
      mouse: {
        tracking: false,
        shiftDragHint: false
      },
      diagnostics: []
    })
    expect(caps.keyboard.pasteShortcutShapes).toEqual(['cmd+v', 'ctrl+shift+v'])
  })

  it('detects local Linux kitty with kitty keyboard encoding and kitty paste shortcut', () => {
    const caps = deriveTerminalCapabilities(
      makeSignals({
        env: {
          KITTY_WINDOW_ID: '1',
          TERM: 'xterm-kitty'
        }
      })
    )

    expect(caps.transport).toBe('local')
    expect(caps.terminalFamily).toBe('kitty')
    expect(caps.keyboard.encoding).toBe('kitty')
    expect(caps.keyboard.pasteShortcutShapes).toEqual(['ctrl+shift+v', 'alt+v', 'ctrl+v'])
    expect(caps.copy.writePath).toBe('native')
    expect(caps.copy.readPath).toBe('native')
    expect(caps.copy.copyOnSelect).toBe(false)
  })

  it('treats SSH plus tmux as tmux transport with stacked layers', () => {
    const caps = deriveTerminalCapabilities(
      makeSignals({
        ssh: {
          hasSshConnection: true,
          hasSshClient: true,
          hasSshTty: true
        },
        multiplexer: {
          tmux: true
        }
      })
    )

    expect(caps.transport).toBe('tmux')
    expect(caps.layers).toEqual(['ssh', 'tmux'])
    expect(caps.terminalFamily).toBe('tmux')
    expect(caps.copy.writePath).toBe('tmux-buffer')
  })

  it('uses osc52 over bare SSH', () => {
    const caps = deriveTerminalCapabilities(
      makeSignals({
        env: {
          TERM: 'xterm-256color'
        },
        ssh: {
          hasSshConnection: true,
          hasSshClient: true,
          hasSshTty: true,
          connection: '1 2 3 4'
        }
      })
    )

    expect(caps.transport).toBe('ssh')
    expect(caps.terminalFamily).toBe('unknown')
    expect(caps.copy.writePath).toBe('osc52')
    expect(caps.copy.readPath).toBe('none')
  })

  it.each([
    ['windows terminal', { env: { WT_SESSION: 'abc' } }, 'windows-terminal'],
    ['Warp on macOS', { env: { TERM_PROGRAM: 'WarpTerminal' } }, 'warp'],
    ['Warp on Linux', { env: { TERM_PROGRAM: 'warp' } }, 'warp'],
    ['zellij', { multiplexer: { zellij: true } }, 'zellij'],
    ['cy', { multiplexer: { cy: true } }, 'cy'],
    ['VS Code terminal', { env: { TERM_PROGRAM: 'vscode' } }, 'vscode-xtermjs'],
    ['VTE terminal', { env: { VTE_VERSION: '7600' } }, 'vte'],
    ['xterm fallback', { env: { TERM: 'xterm-256color' } }, 'xterm'],
    ['unknown terminal', { env: {} }, 'unknown']
  ] as const)('maps %s to %s', (_label, overrides, family) => {
    const caps = deriveTerminalCapabilities(makeSignals(overrides))

    expect(caps.terminalFamily).toBe(family)
  })

  it('uses probe results for bracketed paste, kitty keyboard fallback, OSC52 reads, and terminal version', () => {
    const probe: TerminalProbeResult = {
      bracketedPaste: false,
      kittyKeyboard: true,
      osc52ReadSupported: true,
      xtversionName: 'XTerm 369'
    }

    const caps = deriveTerminalCapabilities(
      makeSignals({
        isStdinTty: false,
        isStdoutTty: false,
        env: {}
      }),
      probe
    )

    expect(caps.paste.bracketedPaste).toBe(false)
    expect(caps.keyboard.encoding).toBe('kitty')
    expect(caps.copy.readPath).toBe('osc52-query')
    expect(caps.terminalVersion).toBe('XTerm 369')
  })

  it('emits diagnostics for missing TERM and SSH without a proper TTY', () => {
    const caps = deriveTerminalCapabilities(
      makeSignals({
        isStdinTty: false,
        isStdoutTty: false,
        ssh: {
          hasSshConnection: true,
          hasSshClient: true,
          hasSshTty: false,
          connection: '1 2 3 4'
        }
      })
    )

    expect(caps.diagnostics).toEqual(expect.arrayContaining(['missing TERM', 'SSH without proper TTY']))
  })
})
