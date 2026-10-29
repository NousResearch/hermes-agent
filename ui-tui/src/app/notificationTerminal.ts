export type NotificationTerminal = 'ghostty' | 'iterm2' | 'kitty'

type TerminalEnvironment = Readonly<Record<string, string | undefined>>

export function notificationTerminalForEnv(env: TerminalEnvironment): NotificationTerminal | null {
  const term = env.TERM?.toLowerCase() ?? ''
  const termProgram = env.TERM_PROGRAM?.toLowerCase() ?? ''
  const lcTerminal = env.LC_TERMINAL?.toLowerCase() ?? ''

  if (env.GHOSTTY_BIN_DIR || env.GHOSTTY_RESOURCES_DIR || term.includes('ghostty') || termProgram.includes('ghostty')) {
    return 'ghostty'
  }

  if (env.KITTY_WINDOW_ID || term.includes('kitty') || termProgram.includes('kitty')) {
    return 'kitty'
  }

  if (
    env.WEZTERM_EXECUTABLE ||
    env.WEZTERM_PANE ||
    termProgram.includes('iterm') ||
    lcTerminal.includes('iterm') ||
    termProgram.includes('wezterm')
  ) {
    return 'iterm2'
  }

  return null
}
