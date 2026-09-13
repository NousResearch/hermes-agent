/**
 * Touch scrolling for the embedded xterm.js chat transcript.
 *
 * xterm.js has no built-in touch scrolling: its viewport only reacts to
 * wheel events and scrollbar drags, so on phones/tablets the transcript
 * cannot be swiped and the thin overlay scrollbar is nearly impossible to
 * grab with a finger. This mirrors the dashboard's wheel handler — vertical
 * one-finger drags become scrollLines() on the browser-side transcript and
 * are never turned into PTY mouse-protocol bytes.
 *
 * Returns a cleanup function that detaches all listeners.
 */

/** Minimal terminal surface this helper needs (keeps it unit-testable). */
export interface TouchScrollTerminal {
  readonly rows: number;
  /** Live terminal element; used to measure the rendered cell height. */
  readonly element: HTMLElement | undefined;
  scrollLines(amount: number): void;
}

/** Movement (px) below which a touch is still a tap, not a scroll. */
export const TOUCH_SCROLL_SLOP_PX = 8;

/** Fallback cell height when the rows element is not measurable yet. */
const FALLBACK_CELL_HEIGHT_PX = 18;

interface TouchLike {
  readonly clientY: number;
}

interface TouchEventLike {
  readonly touches: ArrayLike<TouchLike>;
  preventDefault(): void;
}

export function attachTouchScroll(
  host: EventTarget,
  term: TouchScrollTerminal,
): () => void {
  const state = { active: false, lastY: 0, carryPx: 0 };

  const cellHeightPx = () => {
    const rows = term.element?.querySelector(".xterm-rows");
    return rows && term.rows > 0
      ? rows.clientHeight / term.rows
      : FALLBACK_CELL_HEIGHT_PX;
  };

  const onTouchStart = (ev: Event) => {
    const touches = (ev as unknown as TouchEventLike).touches;
    if (!touches || touches.length !== 1) {
      state.active = false;
      return;
    }
    state.lastY = touches[0].clientY;
    state.carryPx = 0;
  };

  const onTouchMove = (ev: Event) => {
    const tev = ev as unknown as TouchEventLike;
    if (!tev.touches || tev.touches.length !== 1) return;
    const y = tev.touches[0].clientY;
    const dyPx = state.lastY - y; // finger up => scroll down
    if (!state.active) {
      // Stay below the slop threshold so taps still focus the terminal (and
      // bring up the soft keyboard) instead of becoming scrolls.
      if (Math.abs(dyPx) < TOUCH_SCROLL_SLOP_PX) return;
      state.active = true;
    }
    state.lastY = y;
    const cell = cellHeightPx();
    state.carryPx += dyPx;
    const lines = Math.trunc(state.carryPx / cell);
    if (lines !== 0) {
      state.carryPx -= lines * cell;
      term.scrollLines(lines);
    }
    // Block rubber-banding / pull-to-refresh while the transcript is being
    // dragged. Requires passive: false on the listener.
    tev.preventDefault();
  };

  const onTouchEnd = () => {
    state.active = false;
    state.carryPx = 0;
  };

  host.addEventListener("touchstart", onTouchStart, { passive: true });
  host.addEventListener("touchmove", onTouchMove, { passive: false });
  host.addEventListener("touchend", onTouchEnd, { passive: true });
  host.addEventListener("touchcancel", onTouchEnd, { passive: true });

  return () => {
    host.removeEventListener("touchstart", onTouchStart);
    host.removeEventListener("touchmove", onTouchMove);
    host.removeEventListener("touchend", onTouchEnd);
    host.removeEventListener("touchcancel", onTouchEnd);
  };
}
