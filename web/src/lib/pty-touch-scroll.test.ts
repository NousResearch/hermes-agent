import { describe, expect, it, vi } from "vitest";

import {
  attachTouchScroll,
  TOUCH_SCROLL_SLOP_PX,
  type TouchScrollTerminal,
} from "./pty-touch-scroll";

function makeTerm(): TouchScrollTerminal & {
  scrollLines: ReturnType<typeof vi.fn<(amount: number) => void>>;
} {
  return {
    rows: 24,
    element: undefined, // unmeasurable → fallback cell height 18
    scrollLines: vi.fn<(amount: number) => void>(),
  };
}

function touch(type: string, ys: number[]): Event {
  const ev = new Event(type, { cancelable: true });
  Object.defineProperty(ev, "touches", {
    value: ys.map((clientY) => ({ clientY })),
  });
  return ev;
}

describe("attachTouchScroll", () => {
  it("scrolls down when the finger moves up past the slop threshold", () => {
    const host = new EventTarget();
    const term = makeTerm();
    attachTouchScroll(host, term);

    host.dispatchEvent(touch("touchstart", [200]));
    host.dispatchEvent(touch("touchmove", [150])); // 50px up → 2 lines at 18px fallback

    expect(term.scrollLines).toHaveBeenCalledExactlyOnceWith(2);
  });

  it("scrolls up when the finger moves down", () => {
    const host = new EventTarget();
    const term = makeTerm();
    attachTouchScroll(host, term);

    host.dispatchEvent(touch("touchstart", [100]));
    host.dispatchEvent(touch("touchmove", [150])); // 50px down

    expect(term.scrollLines).toHaveBeenCalledExactlyOnceWith(-2);
  });

  it("ignores movement below the slop threshold so taps still focus", () => {
    const host = new EventTarget();
    const term = makeTerm();
    attachTouchScroll(host, term);

    host.dispatchEvent(touch("touchstart", [100]));
    host.dispatchEvent(touch("touchmove", [100 - (TOUCH_SCROLL_SLOP_PX - 1)]));
    host.dispatchEvent(touch("touchend", []));

    expect(term.scrollLines).not.toHaveBeenCalled();
  });

  it("accumulates sub-cell drags across move events (no lost remainder)", () => {
    const host = new EventTarget();
    const term = makeTerm();
    attachTouchScroll(host, term);

    host.dispatchEvent(touch("touchstart", [200]));
    // 8px slop activates, then four 6px drags: 8+6+6+6+6=32px → 1 line (18px)
    host.dispatchEvent(touch("touchmove", [192]));
    host.dispatchEvent(touch("touchmove", [186]));
    host.dispatchEvent(touch("touchmove", [180]));
    host.dispatchEvent(touch("touchmove", [174]));

    expect(term.scrollLines).toHaveBeenCalledExactlyOnceWith(1);
  });

  it("ignores multi-touch gestures", () => {
    const host = new EventTarget();
    const term = makeTerm();
    attachTouchScroll(host, term);

    host.dispatchEvent(touch("touchstart", [200, 300]));
    host.dispatchEvent(touch("touchmove", [100, 200]));

    expect(term.scrollLines).not.toHaveBeenCalled();
  });

  it("preventDefaults active drags (blocks rubber-banding / pull-to-refresh)", () => {
    const host = new EventTarget();
    const term = makeTerm();
    attachTouchScroll(host, term);

    host.dispatchEvent(touch("touchstart", [200]));
    const move = touch("touchmove", [150]);
    host.dispatchEvent(move);

    expect(move.defaultPrevented).toBe(true);
  });

  it("stops scrolling after cleanup", () => {
    const host = new EventTarget();
    const term = makeTerm();
    const detach = attachTouchScroll(host, term);

    detach();
    host.dispatchEvent(touch("touchstart", [200]));
    host.dispatchEvent(touch("touchmove", [100]));

    expect(term.scrollLines).not.toHaveBeenCalled();
  });
});
