type TerminalName = string | null

export function detectTerminal(env: NodeJS.ProcessEnv = process.env): TerminalName {
  if (env.CURSOR_TRACE_ID) {
    return 'cursor'
  }

  // Herdr is a terminal multiplexer with Kitty keyboard protocol support.
  // Its panes intentionally expose a generic TERM, so waiting for TERM or
  // TERM_PROGRAM detection prevents the TUI from requesting modified-key
  // reporting and collapses Shift+Enter to plain Enter. Treat a Herdr pane
  // like tmux for capability negotiation.
  if (env.HERDR_ENV === '1') {
    return 'tmux'
  }

  if (env.TERM === 'xterm-ghostty') {
    return 'ghostty'
  }

  if (env.TERM?.includes('kitty')) {
    return 'kitty'
  }

  if (env.TERM_PROGRAM) {
    return env.TERM_PROGRAM
  }

  if (env.TMUX) {
    return 'tmux'
  }

  if (env.STY) {
    return 'screen'
  }

  if (env.KITTY_WINDOW_ID) {
    return 'kitty'
  }

  if (env.WT_SESSION) {
    return 'windows-terminal'
  }

  return env.TERM ?? null
}

export const env = {
  terminal: detectTerminal()
}

// Terminals known to correctly implement OSC 52 clipboard writes
// (ESC ] 52 ; c ; <b64> BEL/ST — osc() in ink/termio/osc.ts emits BEL
// for most terminals and ST for kitty). When detected, setClipboard() skips the
// native-tool safety net entirely — running wl-copy/xclip/pbcopy in
// parallel with OSC 52 races the terminal's own clipboard write and can
// corrupt it (e.g. wl-copy on Wayland holds the selection in a background
// daemon; stacking two writes within ~30ms triggers a SIGTERM race).
// Intentionally conservative: terminals with known flaky or disabled-by-
// default OSC 52 (iTerm2 disables OSC 52 by default; Alacritty detection
// is unreliable) are not on this list. Users on those terminals keep the
// existing behaviour (native safety net fires alongside OSC 52).
//
// Lives here in utils/env.ts (rather than ink/terminal.ts) so that
// ink/termio/osc.ts can import it without creating a circular dependency:
// ink/terminal.ts already imports `link` from ink/termio/osc.ts.
const OSC52_CAPABLE_TERMINALS = ['ghostty', 'kitty', 'WezTerm', 'windows-terminal', 'vscode']

/** True if this terminal is known to correctly handle OSC 52 clipboard
 *  writes, so setClipboard() can skip the native-tool safety net.
 *  Accepts an optional terminal name for testability; defaults to the
 *  module-level `env.terminal` detected at startup. */
export function supportsOsc52Clipboard(terminal: string | null = env.terminal): boolean {
  return OSC52_CAPABLE_TERMINALS.includes(terminal ?? '')
}
