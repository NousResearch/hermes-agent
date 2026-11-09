/**
 * Which image form this TUI's own terminal paints: `kitty` Unicode
 * placeholders or `unicode` half-blocks. Sent with `image.render` and
 * `pet.cells` because one gateway serves many clients, so its env names no
 * client's terminal. Env vars only, never a terminal query that could hang.
 *
 * Mirrors `agent/terminal_image.py`'s `auto` pick (detect_terminal_graphics,
 * supports_kitty_placeholders, the TMUX check), plus GNU screen and the
 * dashboard's xterm.js, which drop the kitty transmit too.
 */
export type TerminalGraphics = 'kitty' | 'unicode'

const lower = (value?: string) => (value ?? '').trim().toLowerCase()

export const detectTerminalGraphics = (env: NodeJS.ProcessEnv = process.env): TerminalGraphics => {
  const term = lower(env.TERM)
  const program = lower(env.TERM_PROGRAM)

  // VS Code/Cursor don't scrub an inherited KITTY_WINDOW_ID; xterm.js shows tofu.
  if (program === 'vscode' || env.HERMES_PTY_HOST === 'dashboard') {
    return 'unicode'
  }

  // Multiplexers swallow the APC transmit (tmux without allow-passthrough, screen always).
  if (env.TMUX || env.STY) {
    return 'unicode'
  }

  // WezTerm speaks kitty APC but paints placeholders as tofu.
  if (program === 'wezterm' || env.WEZTERM_PANE) {
    return 'unicode'
  }

  const kitty =
    Boolean(env.KITTY_WINDOW_ID || env.GHOSTTY_RESOURCES_DIR) ||
    term.includes('kitty') ||
    term.includes('ghostty') ||
    program === 'ghostty'

  return kitty ? 'kitty' : 'unicode'
}
