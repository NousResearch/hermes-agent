export interface TerminalGrid {
  cols: number;
  rows: number;
}

/**
 * Whether the PTY must be sent a RESIZE after the dashboard terminal refits.
 *
 * The font size is chosen from the host WIDTH alone, so gating the RESIZE on a
 * font change drops every height-only resize: fit() changes `term.rows` while
 * the PTY keeps the old row count, and the TUI redraws against stale
 * dimensions. On iOS Safari that is the common case — the URL bar collapses and
 * expands while scrolling, firing visualViewport resizes at constant width —
 * and it shows up as the chat pane flickering between rendered output and
 * empty black.
 */
export function needsPtyResize(
  before: TerminalGrid,
  after: TerminalGrid,
  fontChanged: boolean,
): boolean {
  return (
    fontChanged || before.cols !== after.cols || before.rows !== after.rows
  );
}
