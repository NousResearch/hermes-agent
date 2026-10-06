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
