/**
 * Convert a vertical one-finger drag into xterm scrollback lines.
 *
 * A finger moving upward reveals newer terminal output, which is positive in
 * xterm's `scrollLines` convention. Small moves below one row are ignored so
 * a tap does not nudge the transcript.
 */
export function touchScrollLines(
  startY: number,
  currentY: number,
  rowHeight: number,
): number {
  if (!Number.isFinite(startY) || !Number.isFinite(currentY) || rowHeight <= 0) {
    return 0;
  }
  const rows = Math.trunc((startY - currentY) / rowHeight);
  return rows || 0;
}
