import supportsHyperlinksLib from 'supports-hyperlinks'

// Additional terminals that support OSC 8 hyperlinks but aren't detected by supports-hyperlinks.
// Checked against both TERM_PROGRAM and LC_TERMINAL (the latter is preserved inside tmux).
export const ADDITIONAL_HYPERLINK_TERMINALS = ['ghostty', 'Hyper', 'kitty', 'alacritty', 'iTerm.app', 'iTerm2']

type EnvLike = Record<string, string | undefined>

type SupportsHyperlinksOptions = {
  env?: EnvLike
  stdoutSupported?: boolean
}

/**
 * Returns whether stdout supports OSC 8 hyperlinks.
 * Extends the supports-hyperlinks library with additional terminal detection.
 * @param options Optional overrides for testing (env, stdoutSupported)
 */
export function supportsHyperlinks(options?: SupportsHyperlinksOptions): boolean {
  const stdoutSupported = options?.stdoutSupported ?? supportsHyperlinksLib.stdout

  if (stdoutSupported) {
    return true
  }

  const env = options?.env ?? process.env

  // Check for additional terminals not detected by supports-hyperlinks
  const termProgram = env['TERM_PROGRAM']

  if (termProgram && ADDITIONAL_HYPERLINK_TERMINALS.includes(termProgram)) {
    return true
  }

  // LC_TERMINAL is set by some terminals (e.g. iTerm2) and preserved inside tmux,
  // where TERM_PROGRAM is overwritten to 'tmux'.
  const lcTerminal = env['LC_TERMINAL']

  if (lcTerminal && ADDITIONAL_HYPERLINK_TERMINALS.includes(lcTerminal)) {
    return true
  }

  // Kitty sets TERM=xterm-kitty
  const term = env['TERM']

  if (term?.includes('kitty')) {
    return true
  }

  return false
}
