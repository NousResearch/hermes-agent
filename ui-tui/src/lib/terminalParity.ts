import { detectVSCodeLikeTerminal, shouldPromptForTerminalSetup } from './terminalSetup.js'

export type MacTerminalHint = {
  key: string
  message: string
  tone: 'info' | 'warn'
}

export type MacTerminalContext = {
  isAppleTerminal: boolean
  isRemote: boolean
  isTmux: boolean
  vscodeLike: null | 'cursor' | 'vscode' | 'windsurf'
}

export function detectMacTerminalContext(env: NodeJS.ProcessEnv = process.env): MacTerminalContext {
  const termProgram = env['TERM_PROGRAM'] ?? ''

  return {
    isAppleTerminal: termProgram === 'Apple_Terminal' || !!env['TERM_SESSION_ID'],
    isRemote: !!(env['SSH_CONNECTION'] || env['SSH_TTY'] || env['SSH_CLIENT']),
    isTmux: !!env['TMUX'],
    vscodeLike: detectVSCodeLikeTerminal(env)
  }
}

export async function terminalParityHints(env: NodeJS.ProcessEnv = process.env): Promise<MacTerminalHint[]> {
  const ctx = detectMacTerminalContext(env)
  const hints: MacTerminalHint[] = []

  if (ctx.vscodeLike && (await shouldPromptForTerminalSetup({ env }))) {
    hints.push({
      key: 'ide-setup',
      tone: 'info',
      message: `Detected ${ctx.vscodeLike} terminal · run /terminal-setup for best Cmd+Enter / undo parity`
    })
  }

  if (ctx.isAppleTerminal) {
    hints.push({
      key: 'apple-terminal',
      tone: 'warn',
      message: 'Apple Terminal detected · use /paste for image-only clipboard fallback, and try Ctrl+A / Ctrl+E / Ctrl+U if Cmd+←/→/⌫ gets rewritten'
    })
  }

  if (ctx.isTmux) {
    hints.push({
      key: 'tmux',
      tone: 'warn',
      message: 'tmux detected · clipboard copy/paste uses passthrough when available; allow-passthrough improves OSC52 reliability'
    })
  }

  if (ctx.isRemote) {
    hints.push({
      key: 'remote',
      tone: 'warn',
      message: 'SSH session detected · text clipboard can bridge via OSC52, but image clipboard and local screenshot paths still depend on the machine running Hermes'
    })
  }

  return hints
}
