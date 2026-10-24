/**
 * Long-press detection for dashboard xterm panes on touch devices.
 *
 * xterm.js paints to a canvas, so a mobile long-press cannot start native text
 * selection. Detect the gesture and hand off to `onLongPress` (the chat opens a
 * plain-text selection overlay). Two signals are used because platforms differ:
 *
 * - a still-finger timer (cancelled by movement past `MOVE_TOLERANCE_PX`, a
 *   second finger, or touchend/touchcancel), and
 * - a touch-originated `contextmenu`, which Android Chrome fires on long-press
 *   (and then cancels the touch, so the timer alone would never fire there).
 *
 * This module never scrolls and never calls `preventDefault()` on touch events,
 * so it composes with `installTerminalTouchScroll`, which owns drag scrolling.
 */

export const LONG_PRESS_MS = 550;
export const MOVE_TOLERANCE_PX = 10;
// A contextmenu this soon after a touchstart is treated as touch-originated.
const TOUCH_CONTEXTMENU_WINDOW_MS = 2000;

const PASSIVE_CAPTURE = { capture: true, passive: true } as const;
const CAPTURE = { capture: true } as const;

export interface TerminalLongPressOptions {
  delayMs?: number;
  /**
   * Checked just before firing; return false to veto, e.g. when the hold is
   * part of a scroll or caught a momentum fling.
   */
  shouldTrigger?: () => boolean;
  /** Only watch touches whose target passes. Default: every touch. */
  acceptTarget?: (target: EventTarget | null) => boolean;
}

export function installTerminalLongPress(
  host: HTMLElement,
  onLongPress: () => void,
  { delayMs = LONG_PRESS_MS, shouldTrigger, acceptTarget }: TerminalLongPressOptions = {},
): () => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let startX = 0;
  let startY = 0;
  let lastTouchAt = 0;
  // The current touch is disqualified (drifted, second finger, or not ours):
  // a later timer or contextmenu for it must not open the overlay.
  let disqualified = true;

  const cancel = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const fire = () => {
    if (disqualified) return;
    if (shouldTrigger && !shouldTrigger()) return;
    onLongPress();
  };

  const onTouchStart = (ev: TouchEvent) => {
    lastTouchAt = Date.now();
    cancel();
    disqualified =
      ev.touches.length !== 1 || (acceptTarget !== undefined && !acceptTarget(ev.target));
    if (disqualified) return;
    startX = ev.touches[0].clientX;
    startY = ev.touches[0].clientY;
    timer = setTimeout(() => {
      timer = null;
      fire();
    }, delayMs);
  };

  const onTouchMove = (ev: TouchEvent) => {
    if (disqualified) return;
    const t = ev.touches[0];
    if (
      ev.touches.length !== 1 ||
      !t ||
      Math.hypot(t.clientX - startX, t.clientY - startY) > MOVE_TOLERANCE_PX
    ) {
      disqualified = true;
      cancel();
    }
  };

  const onContextMenu = (ev: MouseEvent) => {
    const fromTouch =
      Date.now() - lastTouchAt < TOUCH_CONTEXTMENU_WINDOW_MS ||
      (ev as PointerEvent).pointerType === "touch";
    if (!fromTouch) return;
    // Always swallow a touch contextmenu (xterm would treat it as a
    // right-click), but only open the overlay for a genuine still hold.
    ev.preventDefault();
    ev.stopPropagation();
    cancel();
    fire();
    disqualified = true;
  };

  host.addEventListener("touchstart", onTouchStart, PASSIVE_CAPTURE);
  host.addEventListener("touchmove", onTouchMove, PASSIVE_CAPTURE);
  host.addEventListener("touchend", cancel, CAPTURE);
  host.addEventListener("touchcancel", cancel, CAPTURE);
  host.addEventListener("contextmenu", onContextMenu, CAPTURE);

  return () => {
    cancel();
    host.removeEventListener("touchstart", onTouchStart, PASSIVE_CAPTURE);
    host.removeEventListener("touchmove", onTouchMove, PASSIVE_CAPTURE);
    host.removeEventListener("touchend", cancel, CAPTURE);
    host.removeEventListener("touchcancel", cancel, CAPTURE);
    host.removeEventListener("contextmenu", onContextMenu, CAPTURE);
  };
}
