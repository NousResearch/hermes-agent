type TerminalName = string | null

function detectTerminal(): TerminalName {
  if (process.env.CURSOR_TRACE_ID) {
    return 'cursor'
  }

  if (process.env.TERM === 'xterm-ghostty') {
    return 'ghostty'
  }

  if (process.env.TERM?.includes('kitty')) {
    return 'kitty'
  }

  if (process.env.TERM_PROGRAM) {
    return process.env.TERM_PROGRAM
  }

  if (process.env.TMUX) {
    return 'tmux'
  }

  if (process.env.STY) {
    return 'screen'
  }

  if (process.env.KITTY_WINDOW_ID) {
    return 'kitty'
  }

  if (process.env.WT_SESSION) {
    return 'windows-terminal'
  }

  return process.env.TERM ?? null
}

export const env = {
  terminal: detectTerminal()
}
