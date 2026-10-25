import { t } from '../i18n/runtime.js'

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
  options?: { fileOps?: Partial<FileOps>; homeDir?: string; platform?: NodeJS.Platform }
): Promise<MacTerminalHint[]> {
  const ctx = detectMacTerminalContext(env)
  const hints: MacTerminalHint[] = []

  if (
    ctx.vscodeLike &&
    (await shouldPromptForTerminalSetup({
      env,
      fileOps: options?.fileOps,
      homeDir: options?.homeDir,
      platform: options?.platform
    }))
  ) {
    hints.push({
      key: 'ide-setup',
      tone: 'info',
      message: t('libText.terminalParity.ideSetup', ctx.vscodeLike)
    })
  }

  if (ctx.isAppleTerminal) {
    hints.push({
      key: 'apple-terminal',
      tone: 'warn',
      message: t('libText.terminalParity.appleTerminal')
    })
  }

  if (ctx.isTmux) {
    hints.push({
      key: 'tmux',
      tone: 'warn',
      message: t('libText.terminalParity.tmux')
    })
  }

  if (ctx.isRemote) {
    hints.push({
      key: 'remote',
      tone: 'warn',
      message: t('libText.terminalParity.remote')
    })
  }

  return hints
}
