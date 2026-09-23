/** Shift+Up — inner TUI maps this to one line of transcript scroll-up. */
export const SHIFT_UP = "\x1b[1;2A";
/** Shift+Down — inner TUI maps this to one line of transcript scroll-down. */
export const SHIFT_DOWN = "\x1b[1;2B";

/** Browser wheel deltaY units per one transcript row (matches ChatPage history). */
export const WHEEL_DELTA_PER_ROW = 50;
/** Cap rows queued per wheel tick so one trackpad flick cannot flood the PTY. */
export const WHEEL_MAX_ROWS_PER_TICK = 6;

/**
 * Key sequences the embedded Hermes TUI maps to line-by-line transcript
 * scrolling. Returns an empty array when there is nothing to send.
 */
export function wheelScrollSequences(deltaY: number): string[] {
  if (!Number.isFinite(deltaY) || deltaY === 0) {
    return [];
  }

  const rows = Math.min(
    WHEEL_MAX_ROWS_PER_TICK,
    Math.max(1, Math.round(Math.abs(deltaY) / WHEEL_DELTA_PER_ROW)),
  );
  const seq = deltaY > 0 ? SHIFT_DOWN : SHIFT_UP;
  return Array.from({ length: rows }, () => seq);
}
