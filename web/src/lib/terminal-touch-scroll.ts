interface ScrollableTerminal {
  buffer: {
    active: {
      baseY: number;
      viewportY: number;
    };
  };
  scrollLines: (amount: number) => void;
}

const TOUCH_SCROLL_PX_PER_LINE = 18;
const TOUCH_SCROLL_MIN_PX = 2;
const CAPTURE = { capture: true } as const;
const PASSIVE_CAPTURE = { capture: true, passive: true } as const;
const ACTIVE_CAPTURE = { capture: true, passive: false } as const;

function firstTouchY(ev: TouchEvent): number | null {
  const touch = ev.touches[0] ?? ev.changedTouches[0];

  return touch ? touch.clientY : null;
}

function canScrollTerminal(term: ScrollableTerminal, amount: number): boolean {
  const { baseY, viewportY } = term.buffer.active;

  if (amount < 0) {
    return viewportY > 0;
  }

  if (amount > 0) {
    return viewportY < baseY;
  }

  return false;
}

/**
 * Add touch-drag scrollback support for dashboard xterm panes.
 *
 * xterm.js handles wheel/trackpad scrollback for the embedded TUI, but iPadOS
 * WebKit does not translate a finger drag inside the terminal into wheel events
 * or native scrollbar dragging. Mirror the wheel path by converting vertical
 * touch movement into `scrollLines()` while there is xterm scrollback available;
 * at the top/bottom boundary we leave the event alone so the outer dashboard
 * page can still move above the software keyboard.
 */
export function installTerminalTouchScroll(host: HTMLElement, term: ScrollableTerminal): () => void {
  let lastY: number | null = null;
  let residualPx = 0;

  const reset = () => {
    lastY = null;
    residualPx = 0;
  };

  const onTouchStart = (ev: TouchEvent) => {
    if (ev.touches.length !== 1) {
      reset();
      return;
    }

    lastY = firstTouchY(ev);
    residualPx = 0;
  };

  const onTouchMove = (ev: TouchEvent) => {
    if (ev.touches.length !== 1 || lastY === null) {
      reset();
      return;
    }

    const nextY = firstTouchY(ev);
    if (nextY === null) {
      reset();
      return;
    }

    const deltaPx = lastY - nextY;
    lastY = nextY;

    if (Math.abs(deltaPx) < TOUCH_SCROLL_MIN_PX) {
      return;
    }

    residualPx += deltaPx;
    const lines = residualPx / TOUCH_SCROLL_PX_PER_LINE;
    const wholeLines = lines < 0 ? Math.ceil(lines) : Math.floor(lines);

    if (wholeLines === 0) {
      return;
    }

    if (!canScrollTerminal(term, wholeLines)) {
      return;
    }

    residualPx -= wholeLines * TOUCH_SCROLL_PX_PER_LINE;
    term.scrollLines(wholeLines);
    ev.preventDefault();
    ev.stopPropagation();
  };

  host.addEventListener("touchstart", onTouchStart, PASSIVE_CAPTURE);
  host.addEventListener("touchmove", onTouchMove, ACTIVE_CAPTURE);
  host.addEventListener("touchcancel", reset, CAPTURE);
  host.addEventListener("touchend", reset, CAPTURE);

  return () => {
    host.removeEventListener("touchstart", onTouchStart, PASSIVE_CAPTURE);
    host.removeEventListener("touchmove", onTouchMove, ACTIVE_CAPTURE);
    host.removeEventListener("touchcancel", reset, CAPTURE);
    host.removeEventListener("touchend", reset, CAPTURE);
  };
}
