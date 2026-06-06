// Detect whether the current terminal supports any inline image protocol.
// The result is used to decide whether to render image previews or fall back
// to text-only metadata in the chat history.

export type ImageProtocol = 'kitty' | 'iterm' | 'sixel'

export function terminalSupportsImages(): boolean {
  return preferredImageProtocol() !== null
}

export function preferredImageProtocol(): ImageProtocol | null {
  const term = (process.env.TERM ?? '').toLowerCase()
  const termProgram = (process.env.TERM_PROGRAM ?? '').toLowerCase()

  // Kitty Graphics Protocol — best quality, modern.
  if (term.includes('kitty') || termProgram === 'kitty') {
    return 'kitty'
  }

  // iTerm2 inline images — macOS only.
  if (termProgram === 'iterm.app') {
    return 'iterm'
  }

  // Sixel — broadest hardware support (xterm with -ti flag, mlterm, WezTerm, foot).
  if (
    term.includes('mlterm') ||
    term.includes('wezterm') ||
    term.includes('foot') ||
    (term.includes('xterm') && term.includes('sixel'))
  ) {
    return 'sixel'
  }

  return null
}
