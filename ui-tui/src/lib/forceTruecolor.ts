/**
 * Force 24-bit truecolor output before any chalk / supports-color import.
 *
 * Why this exists:
 *   The base CLI (Python/Rich) emits banner colors as truecolor ANSI
 *   (`\033[38;2;R;G;Bm`). The TUI renders through Ink → chalk, whose
 *   supports-color auto-detection defaults to 256-color on macOS Terminal.app
 *   and any terminal that does NOT set `COLORTERM=truecolor`. In 256-color
 *   mode, chalk downsamples `#FFD700` (gold) and `#FFBF00` (amber) to the
 *   *same* xterm-256 palette slot (220) — collapsing the banner gradient
 *   into a single flat yellow band. The bronze and dim rows also lose
 *   contrast against each other.
 *
 *   Terminal.app (macOS 12+), iTerm2, kitty, Alacritty, VS Code, Cursor,
 *   and WezTerm all render truecolor correctly. The few that don't
 *   (ancient xterm, some CI environments) can set `HERMES_TUI_TRUECOLOR=0`
 *   to opt out.
 *
 * This MUST run before any `chalk` or `supports-color` import. supports-color
 * caches its level on first load, so nudging env vars after that point has
 * no effect.
 */

if (process.env.HERMES_TUI_TRUECOLOR !== '0' && !process.env.NO_COLOR && !process.env.FORCE_COLOR) {
  if (!process.env.COLORTERM) {
    process.env.COLORTERM = 'truecolor'
  }

  process.env.FORCE_COLOR = '3'
}

export {}
