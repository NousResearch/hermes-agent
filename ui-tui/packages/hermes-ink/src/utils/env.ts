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
