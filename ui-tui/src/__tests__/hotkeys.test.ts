import { describe, expect, it } from 'vitest'

import { collectTerminalSignals } from '../lib/terminalSignals.js'
import { deriveTerminalCapabilities } from '../lib/terminalCapabilities.js'
import { buildHelpHintHotkeys } from '../content/hotkeys.js'

describe('buildHelpHintHotkeys', () => {
  it('shows Ctrl+Shift+V for modern non-mac terminals', () => {
    const signals = collectTerminalSignals({ env: { TERM: 'wezterm', WEZTERM_PANE: '1' }, platform: 'linux' })
    const caps = deriveTerminalCapabilities(signals)

    const rows = buildHelpHintHotkeys({ signals, capabilities: caps })

    expect(rows.some(([key]) => key.startsWith('Ctrl+Shift+V'))).toBe(true)
  })

  it('shows Cmd+V for mac terminals', () => {
    const signals = collectTerminalSignals({
      env: { TERM_PROGRAM: 'iTerm.app', SHELL: '/bin/zsh', ZSH_VERSION: '5.9' },
      platform: 'darwin',
      shellExecutable: '/bin/zsh',
      shellArgv0: '-zsh'
    })
    const caps = deriveTerminalCapabilities(signals)

    const rows = buildHelpHintHotkeys({ signals, capabilities: caps })

    expect(rows.some(([key]) => key.startsWith('Cmd+V'))).toBe(true)
  })

  it('shows shift-drag and tmux copy hints when relevant', () => {
    const signals = collectTerminalSignals({ env: { TMUX: '/tmp/tmux,1,0', TERM: 'screen-256color' }, platform: 'linux' })
    const caps = deriveTerminalCapabilities(signals)

    const rows = buildHelpHintHotkeys({ signals, capabilities: caps })

    expect(rows.some(([key]) => key === 'Shift-drag')).toBe(true)
    expect(rows.some(([key, text]) => key.includes('tmux') || text.includes('tmux'))).toBe(true)
  })
})
