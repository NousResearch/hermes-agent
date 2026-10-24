interface ScrollableTerminal {
  buffer: {
    active: {
      baseY: number;
      viewportY: number;
    };
  };
  scrollLines: (amount: number) => void;
}

export interface TerminalTouchScrollOptions {
  /**
   * Finger travel per scrolled line. A function is re-read at each touchstart
   * so callers can follow the live cell height (font-size changes, refits).
   * Default 18 px.
   */
  pxPerLine?: number | (() => number);
  /**
   * Own every one-finger drag that starts on the terminal, tracking the finger
   * 1:1: each touchmove is `preventDefault()`ed (so the outer page never pans,
   * rubber-bands or pull-to-refreshes underneath), every pixel of movement is
   * counted (no small-move dead zone, so slow drags don't lag), and travel past
   * the top/bottom of scrollback is discarded rather than banked (so reversing
   * at a boundary responds immediately). Default false: the boundary
   * fall-through described below.
   */
  containGesture?: boolean;
  /**
   * Decide at touchstart whether this gesture is ours (e.g. only touches on
   * the terminal itself, not overlays/toolbars sharing the listener element).
   * Default: every touch.
   */
  acceptTarget?: (target: EventTarget | null) => boolean;
}

/** Cleanup function, plus a probe other touch handlers can coordinate with. */
export type TerminalTouchScrollHandle = (() => void) & {
  /** True once the current gesture has scrolled at least one line. */
  isScrolling: () => boolean;
};

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
 * page can still move above the software keyboard (unless `containGesture`).
 */
export function installTerminalTouchScroll(
  host: HTMLElement,
  term: ScrollableTerminal,
  options: TerminalTouchScrollOptions = {},
): TerminalTouchScrollHandle {
  const { containGesture = false, acceptTarget } = options;
  const readPxPerLine = (): number => {
    const raw =
      typeof options.pxPerLine === "function" ? options.pxPerLine() : options.pxPerLine;
    return raw && Number.isFinite(raw) && raw > 0 ? raw : TOUCH_SCROLL_PX_PER_LINE;
  };

  let lastY: number | null = null;
  let residualPx = 0;
  let pxPerLine = readPxPerLine();
  let gestureScrolled = false;

  const reset = () => {
    lastY = null;
    residualPx = 0;
  };

  const onTouchStart = (ev: TouchEvent) => {
    gestureScrolled = false;
    if (ev.touches.length !== 1 || (acceptTarget && !acceptTarget(ev.target))) {
      reset();
      return;
    }

    lastY = firstTouchY(ev);
    residualPx = 0;
    pxPerLine = readPxPerLine();
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

    if (containGesture && ev.cancelable) {
      ev.preventDefault();
    }

    const deltaPx = lastY - nextY;
    lastY = nextY;

    if (!containGesture && Math.abs(deltaPx) < TOUCH_SCROLL_MIN_PX) {
      return;
    }

    residualPx += deltaPx;
    const lines = residualPx / pxPerLine;
    const wholeLines = lines < 0 ? Math.ceil(lines) : Math.floor(lines);

    if (wholeLines === 0) {
      return;
    }

    if (!canScrollTerminal(term, wholeLines)) {
      if (containGesture) {
        residualPx = 0;
      }
      return;
    }

    residualPx -= wholeLines * pxPerLine;
    term.scrollLines(wholeLines);
    gestureScrolled = true;
    ev.preventDefault();
    ev.stopPropagation();
  };

  host.addEventListener("touchstart", onTouchStart, PASSIVE_CAPTURE);
  host.addEventListener("touchmove", onTouchMove, ACTIVE_CAPTURE);
  host.addEventListener("touchcancel", reset, CAPTURE);
  host.addEventListener("touchend", reset, CAPTURE);

  const cleanup = () => {
    host.removeEventListener("touchstart", onTouchStart, PASSIVE_CAPTURE);
    host.removeEventListener("touchmove", onTouchMove, ACTIVE_CAPTURE);
    host.removeEventListener("touchcancel", reset, CAPTURE);
    host.removeEventListener("touchend", reset, CAPTURE);
  };

  return Object.assign(cleanup, {
    isScrolling: () => gestureScrolled,
  });
}
