import { type Locale, translate } from '../i18n/index.js'

import {
  detectVSCodeLikeTerminal,
  type FileOps,
  isRemoteShellSession,
  shouldPromptForTerminalSetup
} from './terminalSetup.js'

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
    isRemote: isRemoteShellSession(env),
    isTmux: !!env['TMUX'],
    vscodeLike: detectVSCodeLikeTerminal(env)
  }
}

export async function terminalParityHints(
  env: NodeJS.ProcessEnv = process.env,
  options?: { fileOps?: Partial<FileOps>; homeDir?: string; locale?: Locale }
): Promise<MacTerminalHint[]> {
  const ctx = detectMacTerminalContext(env)
  const hints: MacTerminalHint[] = []
  const locale = options?.locale ?? 'en'

  if (
    ctx.vscodeLike &&
    (await shouldPromptForTerminalSetup({ env, fileOps: options?.fileOps, homeDir: options?.homeDir }))
  ) {
    hints.push({
      key: 'ide-setup',
      tone: 'info',
      message: translate(locale, 'terminal.hint.ide', { terminal: ctx.vscodeLike })
    })
  }

  if (ctx.isAppleTerminal) {
    hints.push({
      key: 'apple-terminal',
      tone: 'warn',
      message: translate(locale, 'terminal.hint.apple')
    })
  }

  if (ctx.isTmux) {
    hints.push({
      key: 'tmux',
      tone: 'warn',
      message: translate(locale, 'terminal.hint.tmux')
    })
  }

  if (ctx.isRemote) {
    hints.push({
      key: 'remote',
      tone: 'warn',
      message: translate(locale, 'terminal.hint.remote')
    })
  }

  return hints
}
