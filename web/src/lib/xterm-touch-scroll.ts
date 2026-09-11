/**
 * Finger-drag scrolling for the dashboard's xterm-based chat transcript.
 *
 * xterm.js paints its text into `.xterm-screen`, which — since xterm 6 — is a
 * sibling of the legacy `.xterm-viewport` node and sits *on top* of it. The
 * real scroll position lives in xterm's own VS Code-derived ScrollableElement,
 * which only listens for wheel events. So on a tablet a finger drag over the
 * transcript reaches no native scroller at all: the only way to move the
 * backlog was to grab the thin scrollbar on the right edge, while a mouse
 * wheel worked because ChatPage maps wheel deltas onto `term.scrollLines()`.
 *
 * This module gives touch the same treatment: a one-finger vertical drag is
 * translated into terminal lines (at a slight gain, because line-quantised
 * text feels sluggish at exact 1:1), and a flick releases into a ~0.55 s
 * momentum glide. Taps, long-presses (text selection) and two-finger gestures
 * (pinch-zoom) are deliberately left alone — nothing is consumed until the
 * pointer has travelled far enough to be unambiguously a vertical scroll, and
 * drags that start on the scrollbar are left to xterm's own scrollbar.
 */

import type { Terminal } from "@xterm/xterm";

/** Finger travel (px) before a touch becomes a scroll instead of a tap. */
export const DRAG_START_THRESHOLD_PX = 6;
/**
 * How far the transcript moves per pixel of finger travel. 1 = exact
 * tracking; line-quantised text reads as "dragging through treacle" at 1:1,
 * and 1.8 is where a tablet stops feeling like the text lags behind the
 * finger. Pass `gain: 1` for exact tracking.
 */
export const DEFAULT_SCROLL_GAIN = 1.8;
/** Minimum release speed (px/ms) that turns into a momentum scroll. */
export const FLICK_MIN_VELOCITY_PX_PER_MS = 0.08;
/** Momentum decay per millisecond (≈0.94 per 16 ms frame ⇒ ~0.55 s glide). */
export const MOMENTUM_DECAY_PER_MS = 0.996;
/** Speed (px/ms) below which momentum stops. */
export const MOMENTUM_STOP_VELOCITY_PX_PER_MS = 0.015;
/** Rolling window (ms) used to estimate release velocity. */
export const VELOCITY_WINDOW_MS = 90;
/** Touch points closer than this to the right edge belong to the scrollbar. */
const SCROLLBAR_SLACK_PX = 4;

/**
 * Turns pixel travel into whole terminal lines, keeping the sub-line remainder
 * so that slow drags still advance instead of being rounded away.
 */
export class TouchScrollAccumulator {
  private residual = 0;
  private lineHeight: number;

  constructor(lineHeightPx: number) {
    this.lineHeight = lineHeightPx > 0 ? lineHeightPx : 1;
  }

  setLineHeight(lineHeightPx: number): void {
    this.lineHeight = lineHeightPx > 0 ? lineHeightPx : 1;
  }

  reset(): void {
    this.residual = 0;
  }

  /**
   * Feed finger travel. Positive = the finger moved up, so the transcript
   * should move towards newer output. Returns whole lines to scroll now.
   */
  push(deltaPx: number): number {
    if (!Number.isFinite(deltaPx) || deltaPx === 0) {
      return 0;
    }
    this.residual += deltaPx;
    const lines = Math.trunc(this.residual / this.lineHeight);
    if (lines !== 0) {
      this.residual -= lines * this.lineHeight;
    }
    return lines;
  }
}

/** Speed of the next momentum frame, decayed by the elapsed time. */
export function decayMomentum(
  velocityPxPerMs: number,
  dtMs: number,
  decayPerMs: number = MOMENTUM_DECAY_PER_MS,
): number {
  if (!(dtMs > 0)) {
    return velocityPxPerMs;
  }
  return velocityPxPerMs * Math.pow(decayPerMs, dtMs);
}

/** Coerce a caller-supplied gain into a usable value (falls back to 1.5). */
export function resolveScrollGain(gain?: number): number {
  return typeof gain === "number" && Number.isFinite(gain) && gain > 0
    ? gain
    : DEFAULT_SCROLL_GAIN;
}

/** True when the touch landed on xterm's own scrollbar / right edge gutter. */
export function isScrollbarTouch(
  host: HTMLElement,
  clientX: number,
): boolean {
  const screen = host.querySelector<HTMLElement>(".xterm-screen");
  if (!screen) {
    return false;
  }
  const rect = screen.getBoundingClientRect();
  if (rect.width <= 0) {
    return false;
  }
  return clientX >= rect.right - SCROLLBAR_SLACK_PX;
}

export interface XtermTouchScrollHandle {
  dispose(): void;
}

export interface XtermTouchScrollOptions {
  /** Override the measured cell height, e.g. for tests. */
  lineHeightPx?: () => number;
  /**
   * Transcript pixels per finger pixel (default {@link DEFAULT_SCROLL_GAIN}).
   * Raise it for a faster feel, set 1 for exact 1:1 tracking.
   */
  gain?: number;
}

/**
 * Wire finger-drag scrolling onto an opened xterm terminal. Attach to the
 * container element that holds the terminal (events bubble up from the
 * screen), and dispose it together with the terminal.
 */
export function attachXtermTouchScroll(
  term: Terminal,
  host: HTMLElement,
  options: XtermTouchScrollOptions = {},
): XtermTouchScrollHandle {
  const measureLineHeight = (): number => {
    if (options.lineHeightPx) {
      return options.lineHeightPx();
    }
    // The DOM renderer sizes every row element to the cell height the terminal
    // itself scrolls by — the most faithful pixel-per-line source when it is
    // available (the canvas/WebGL renderers do not emit row elements).
    const row = host.querySelector<HTMLElement>(".xterm-rows > div");
    const rowHeight = row?.getBoundingClientRect().height ?? 0;
    if (rowHeight > 0) {
      return rowHeight;
    }
    const rows = term.rows > 0 ? term.rows : 1;
    const screen = host.querySelector<HTMLElement>(".xterm-screen");
    const height = screen?.getBoundingClientRect().height ?? 0;
    return height > 0 ? height / rows : 0;
  };

  const lines = new TouchScrollAccumulator(measureLineHeight());
  const gain = resolveScrollGain(options.gain);
  let identifier: number | null = null;
  let active = false;
  let dragging = false;
  let startX = 0;
  let startY = 0;
  let lastY = 0;
  let samples: { t: number; y: number }[] = [];
  let velocity = 0;
  let momentum = 0;
  let rafId = 0;
  let lastFrameT = 0;

  const scrollByLines = (count: number): void => {
    if (count !== 0) {
      term.scrollLines(count);
    }
  };

  /** Feed finger travel through the gain, hand whole lines to xterm. */
  const pushTravel = (deltaPx: number): void => {
    scrollByLines(lines.push(deltaPx * gain));
  };

  const stopMomentum = (): void => {
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
    momentum = 0;
  };

  const momentumFrame = (timeStamp: number): void => {
    const dt = lastFrameT > 0 ? Math.min(48, timeStamp - lastFrameT) : 16;
    lastFrameT = timeStamp;
    pushTravel(momentum * dt);
    momentum = decayMomentum(momentum, dt);
    if (Math.abs(momentum) <= MOMENTUM_STOP_VELOCITY_PX_PER_MS) {
      rafId = 0;
      momentum = 0;
      return;
    }
    rafId = requestAnimationFrame(momentumFrame);
  };

  const startMomentum = (v: number): void => {
    stopMomentum();
    if (Math.abs(v) < FLICK_MIN_VELOCITY_PX_PER_MS) {
      lines.reset();
      return;
    }
    momentum = v;
    lastFrameT = 0;
    rafId = requestAnimationFrame(momentumFrame);
  };

  const endGesture = (withMomentum: boolean): void => {
    const wasDragging = dragging;
    const releaseVelocity = velocity;
    active = false;
    dragging = false;
    identifier = null;
    samples = [];
    velocity = 0;
    if (wasDragging && withMomentum) {
      startMomentum(releaseVelocity);
    } else {
      stopMomentum();
      lines.reset();
    }
  };

  const touchById = (
    touches: TouchList,
    id: number | null,
  ): Touch | null => {
    if (id === null) {
      return null;
    }
    for (let i = 0; i < touches.length; i += 1) {
      const touch = touches.item(i);
      if (touch && touch.identifier === id) {
        return touch;
      }
    }
    return null;
  };

  const onTouchStart = (event: TouchEvent): void => {
    // A new finger cancels any in-flight flick so the transcript stops where
    // the user put it.
    stopMomentum();
    active = false;
    dragging = false;
    identifier = null;
    velocity = 0;
    samples = [];
    if (event.touches.length !== 1) {
      // Pinch-zoom / multi-finger gestures are not ours to consume.
      return;
    }
    const touch = event.touches[0];
    if (!touch || isScrollbarTouch(host, touch.clientX)) {
      return;
    }
    identifier = touch.identifier;
    active = true;
    startX = touch.clientX;
    startY = touch.clientY;
    lastY = touch.clientY;
    samples = [{ t: event.timeStamp || performance.now(), y: touch.clientY }];
    lines.setLineHeight(measureLineHeight());
    lines.reset();
  };

  const onTouchMove = (event: TouchEvent): void => {
    if (!active) {
      return;
    }
    if (event.touches.length > 1) {
      // Escalate to the browser (pinch-zoom); we must not have swallowed the
      // gesture, and by now we may have — so stop scrolling and bail out.
      active = false;
      dragging = false;
      identifier = null;
      lines.reset();
      return;
    }
    const touch = touchById(event.touches, identifier);
    if (!touch) {
      return;
    }
    const y = touch.clientY;
    if (!dragging) {
      const travelY = startY - y;
      if (Math.abs(travelY) < DRAG_START_THRESHOLD_PX) {
        // Still a candidate tap / long-press: don't consume the event.
        return;
      }
      if (Math.abs(travelY) < Math.abs(startX - touch.clientX)) {
        // Horizontal intent (selection drag, page gesture) — not ours.
        active = false;
        identifier = null;
        return;
      }
      dragging = true;
      lines.reset();
    }
    const deltaPx = lastY - y;
    lastY = y;
    event.preventDefault();
    pushTravel(deltaPx);

    const timeStamp = event.timeStamp || performance.now();
    samples.push({ t: timeStamp, y });
    while (samples.length > 2 && timeStamp - samples[0].t > VELOCITY_WINDOW_MS) {
      samples.shift();
    }
    const first = samples[0];
    const dt = timeStamp - first.t;
    velocity = dt > 0 ? (first.y - y) / dt : 0;
  };

  const onTouchEnd = (): void => {
    if (!active) {
      return;
    }
    endGesture(true);
  };

  const onTouchCancel = (): void => {
    if (!active) {
      return;
    }
    endGesture(false);
  };

  const onWheel = (): void => {
    // Wheel scrolling is handled by ChatPage; a running flick would fight it.
    stopMomentum();
  };

  host.addEventListener("touchstart", onTouchStart, { passive: true });
  host.addEventListener("touchmove", onTouchMove, { passive: false });
  host.addEventListener("touchend", onTouchEnd, { passive: true });
  host.addEventListener("touchcancel", onTouchCancel, { passive: true });
  host.addEventListener("wheel", onWheel, { passive: true, capture: true });

  return {
    dispose(): void {
      stopMomentum();
      host.removeEventListener("touchstart", onTouchStart);
      host.removeEventListener("touchmove", onTouchMove);
      host.removeEventListener("touchend", onTouchEnd);
      host.removeEventListener("touchcancel", onTouchCancel);
      host.removeEventListener("wheel", onWheel, true);
    },
  };
}
