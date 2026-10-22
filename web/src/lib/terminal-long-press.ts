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

export const LONG_PRESS_MS = 450;
export const MOVE_TOLERANCE_PX = 10;
// A contextmenu this soon after a touchstart is treated as touch-originated.
const TOUCH_CONTEXTMENU_WINDOW_MS = 2000;

const PASSIVE_CAPTURE = { capture: true, passive: true } as const;
const CAPTURE = { capture: true } as const;

export function installTerminalLongPress(
  host: HTMLElement,
  onLongPress: () => void,
  { delayMs = LONG_PRESS_MS }: { delayMs?: number } = {},
): () => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let startX = 0;
  let startY = 0;
  let lastTouchAt = 0;

  const cancel = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const onTouchStart = (ev: TouchEvent) => {
    lastTouchAt = Date.now();
    cancel();
    if (ev.touches.length !== 1) return;
    startX = ev.touches[0].clientX;
    startY = ev.touches[0].clientY;
    timer = setTimeout(() => {
      timer = null;
      onLongPress();
    }, delayMs);
  };

  const onTouchMove = (ev: TouchEvent) => {
    if (timer === null) return;
    const t = ev.touches[0];
    if (
      ev.touches.length !== 1 ||
      !t ||
      Math.hypot(t.clientX - startX, t.clientY - startY) > MOVE_TOLERANCE_PX
    ) {
      cancel();
    }
  };

  const onContextMenu = (ev: MouseEvent) => {
    const fromTouch =
      Date.now() - lastTouchAt < TOUCH_CONTEXTMENU_WINDOW_MS ||
      (ev as PointerEvent).pointerType === "touch";
    if (!fromTouch) return;
    ev.preventDefault();
    ev.stopPropagation();
    cancel();
    onLongPress();
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
