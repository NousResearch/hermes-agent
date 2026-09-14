/**
 * One-finger pan → Hermes Ink transcript scroll, one row at a time.
 *
 * iPhone Safari also synthesizes wheel events and click/caret SGR from the
 * same finger. Callers must ignore wheel on coarse pointers, scroll only
 * after the pan threshold, and drop the click that follows a pan.
 */

export const TOUCH_PAN_PX = 14;

export function touchLineTravel(rowHeight: number): number {
  if (!Number.isFinite(rowHeight) || rowHeight <= 0) return 28;
  return Math.min(36, Math.max(24, rowHeight * 1.25));
}

export function isTouchPan(originY: number, currentY: number): boolean {
  if (!Number.isFinite(originY) || !Number.isFinite(currentY)) return false;
  return Math.abs(currentY - originY) >= TOUCH_PAN_PX;
}

export function touchScrollLines(
  startY: number,
  currentY: number,
  rowHeight: number,
): number {
  if (!Number.isFinite(startY) || !Number.isFinite(currentY) || rowHeight <= 0) {
    return 0;
  }
  const travel = touchLineTravel(rowHeight);
  const rows = Math.trunc((startY - currentY) / travel);
  if (!rows) return 0;
  return rows > 0 ? 1 : -1;
}

/** Keep the remaining sub-line pixels; move the sample toward the finger. */
export function advanceTouchAnchor(startY: number, lines: number, travel: number): number {
  return startY - lines * travel;
}

/** Pixel-wheel tick. Coarse pointers should not use this path at all. */
export function wheelScrollLines(deltaY: number): number {
  if (!Number.isFinite(deltaY) || deltaY === 0) {
    return 0;
  }
  return deltaY > 0 ? 1 : -1;
}

/**
 * Real SGR wheel reports. Hermes Ink parses these as wheelup/wheeldown keys,
 * which bypasses the focused composer and reaches the transcript scroll
 * handler. This is deliberately not PageUp/PageDown: a finger drag should
 * behave like one wheel step, not like a full-page keyboard jump.
 */
export const PTY_WHEEL_UP = "\x1b[<64;1;1M";
export const PTY_WHEEL_DOWN = "\x1b[<65;1;1M";

export function ptyWheelSequence(lines: number): string | null {
  if (!Number.isFinite(lines) || lines === 0) return null;
  return lines > 0 ? PTY_WHEEL_UP : PTY_WHEEL_DOWN;
}
