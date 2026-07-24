/**
 * Coordinate mapping for mouse events in the INLINE (non-alt-screen) TUI.
 *
 * In alt-screen mode the rendered frame fills the viewport, so a terminal
 * mouse cell maps 1:1 to a screen-buffer cell and hit-testing needs no
 * translation. Inline mode is different: the frame is rendered into the
 * primary buffer and is bottom-anchored — its last row sits on the
 * terminal's bottom row and it occupies the last `frameHeight` rows. Rows
 * above the frame have scrolled into the host terminal's native scrollback
 * (committed history) and are no longer part of the live DOM.
 *
 * `resolveInlineMouseTarget` turns an absolute terminal cell into a
 * frame-local screen cell for that bottom-anchored frame, or returns null
 * when the cell isn't addressable. The translation
 *
 *     localRow = absRow - (terminalRows - frameHeight)
 *
 * is exact whether the frame fits the viewport (top row at
 * `terminalRows - frameHeight`) or overflows it (top row above the viewport,
 * so `terminalRows - frameHeight` is negative and the visible rows still map
 * correctly). Columns pass through — inline frames start at column 0.
 */

export interface InlineFrameGeometry {
  /** Total visible terminal rows (`stdout.rows`). */
  readonly terminalRows: number
  /** Width of the current inline frame's screen buffer, in cells. */
  readonly frameWidth: number
  /** Height of the current inline frame's screen buffer, in rows. */
  readonly frameHeight: number
}

export interface InlineMouseTarget {
  readonly col: number
  readonly row: number
}

/**
 * Map an absolute terminal `(col, row)` (0-indexed) to a frame-local screen
 * cell for the bottom-anchored inline frame, or null when the point isn't
 * addressable.
 *
 * `clamp = false` (press / click): reject any point outside the live frame.
 * A click in the scrollback above the frame — or past its right edge — must
 * not resolve to a stale DOM rect and move the caret to the wrong place.
 *
 * `clamp = true` (drag / release while a press is captured): clamp to the
 * frame edges instead of rejecting, so a drag that runs past the top or
 * sides keeps extending the selection to the nearest cell, matching how
 * native terminals treat a drag that leaves the text region.
 */
export function resolveInlineMouseTarget(
  col: number,
  row: number,
  geom: InlineFrameGeometry,
  clamp = false
): InlineMouseTarget | null {
  const { terminalRows, frameWidth, frameHeight } = geom

  if (frameHeight <= 0 || frameWidth <= 0) {
    return null
  }

  const localRow = row - (terminalRows - frameHeight)

  if (clamp) {
    return {
      col: Math.max(0, Math.min(frameWidth - 1, col)),
      row: Math.max(0, Math.min(frameHeight - 1, localRow))
    }
  }

  if (localRow < 0 || localRow >= frameHeight || col < 0 || col >= frameWidth) {
    return null
  }

  return { col, row: localRow }
}
