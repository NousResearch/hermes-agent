import { describe, expect, it, vi } from "vitest";

import { installTerminalTouchScroll } from "./terminal-touch-scroll";

function touchEvent(type: string, y: number): TouchEvent {
  const ev = new Event(type, { bubbles: true, cancelable: true }) as TouchEvent;
  const touch = { clientY: y } as Touch;

  Object.defineProperty(ev, "touches", { value: type === "touchend" ? [] : [touch] });
  Object.defineProperty(ev, "changedTouches", { value: [touch] });

  return ev;
}

function term(viewportY: number, baseY: number) {
  const terminal = {
    buffer: { active: { baseY, viewportY } },
    scrollLines: vi.fn((amount: number) => {
      const active = terminal.buffer.active;
      active.viewportY = Math.max(0, Math.min(active.baseY, active.viewportY + amount));
    }),
  };

  return terminal;
}

describe("installTerminalTouchScroll", () => {
  it("converts one-finger vertical drags into xterm scrollback lines", () => {
    const host = new EventTarget() as HTMLElement;
    const terminal = term(20, 100);
    const cleanup = installTerminalTouchScroll(host, terminal);

    host.dispatchEvent(touchEvent("touchstart", 200));
    const move = touchEvent("touchmove", 164);
    host.dispatchEvent(move);

    expect(terminal.scrollLines).toHaveBeenCalledWith(2);
    expect(move.defaultPrevented).toBe(true);

    cleanup();
  });

  it("lets the outer page handle touch drags at the terminal boundary", () => {
    const host = new EventTarget() as HTMLElement;
    const terminal = term(0, 100);
    const cleanup = installTerminalTouchScroll(host, terminal);

    host.dispatchEvent(touchEvent("touchstart", 100));
    const move = touchEvent("touchmove", 136);
    host.dispatchEvent(move);

    expect(terminal.scrollLines).not.toHaveBeenCalled();
    expect(move.defaultPrevented).toBe(false);

    cleanup();
  });

  it("removes listeners on cleanup", () => {
    const host = new EventTarget() as HTMLElement;
    const terminal = term(20, 100);
    const cleanup = installTerminalTouchScroll(host, terminal);

    cleanup();

    host.dispatchEvent(touchEvent("touchstart", 200));
    host.dispatchEvent(touchEvent("touchmove", 164));

    expect(terminal.scrollLines).not.toHaveBeenCalled();
  });

  describe("options", () => {
    it("uses a custom px-per-line, re-read at each touchstart", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(20, 100);
      let cell = 10;
      const cleanup = installTerminalTouchScroll(host, terminal, { pxPerLine: () => cell });

      host.dispatchEvent(touchEvent("touchstart", 200));
      host.dispatchEvent(touchEvent("touchmove", 170));
      expect(terminal.scrollLines).toHaveBeenLastCalledWith(3);

      cell = 15;
      host.dispatchEvent(touchEvent("touchend", 170));
      host.dispatchEvent(touchEvent("touchstart", 200));
      host.dispatchEvent(touchEvent("touchmove", 170));
      expect(terminal.scrollLines).toHaveBeenLastCalledWith(2);

      cleanup();
    });

    it("containGesture blocks the page on sub-line moves and at the boundary", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(0, 100);
      const cleanup = installTerminalTouchScroll(host, terminal, { containGesture: true });

      host.dispatchEvent(touchEvent("touchstart", 100));
      const small = touchEvent("touchmove", 95);
      host.dispatchEvent(small);
      const atTop = touchEvent("touchmove", 140);
      host.dispatchEvent(atTop);

      expect(terminal.scrollLines).not.toHaveBeenCalled();
      expect(small.defaultPrevented).toBe(true);
      expect(atTop.defaultPrevented).toBe(true);

      cleanup();
    });

    it("default mode still lets sub-line moves through to the page", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(20, 100);
      const cleanup = installTerminalTouchScroll(host, terminal);

      host.dispatchEvent(touchEvent("touchstart", 100));
      const small = touchEvent("touchmove", 95);
      host.dispatchEvent(small);

      expect(small.defaultPrevented).toBe(false);
      cleanup();
    });

    it("containGesture counts small moves so slow drags track the finger", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(20, 100);
      const cleanup = installTerminalTouchScroll(host, terminal, {
        containGesture: true,
        pxPerLine: 10,
      });

      host.dispatchEvent(touchEvent("touchstart", 200));
      for (let y = 199; y >= 190; y -= 1) {
        host.dispatchEvent(touchEvent("touchmove", y));
      }
      expect(terminal.scrollLines).toHaveBeenCalledWith(1);

      cleanup();
    });

    it("default mode keeps the small-move dead zone", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(20, 100);
      const cleanup = installTerminalTouchScroll(host, terminal, { pxPerLine: 10 });

      host.dispatchEvent(touchEvent("touchstart", 200));
      for (let y = 199; y >= 190; y -= 1) {
        host.dispatchEvent(touchEvent("touchmove", y));
      }
      expect(terminal.scrollLines).not.toHaveBeenCalled();

      cleanup();
    });

    it("containGesture discards travel past a boundary so reversing responds at once", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(0, 100);
      const cleanup = installTerminalTouchScroll(host, terminal, {
        containGesture: true,
        pxPerLine: 10,
      });

      host.dispatchEvent(touchEvent("touchstart", 100));
      host.dispatchEvent(touchEvent("touchmove", 300));
      expect(terminal.scrollLines).not.toHaveBeenCalled();

      host.dispatchEvent(touchEvent("touchmove", 290));
      expect(terminal.scrollLines).toHaveBeenCalledWith(1);

      cleanup();
    });

    it("acceptTarget skips gestures that start on excluded targets", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(20, 100);
      let allow = false;
      const cleanup = installTerminalTouchScroll(host, terminal, {
        containGesture: true,
        acceptTarget: () => allow,
      });

      host.dispatchEvent(touchEvent("touchstart", 200));
      const move = touchEvent("touchmove", 150);
      host.dispatchEvent(move);
      expect(terminal.scrollLines).not.toHaveBeenCalled();
      expect(move.defaultPrevented).toBe(false);

      allow = true;
      host.dispatchEvent(touchEvent("touchstart", 200));
      host.dispatchEvent(touchEvent("touchmove", 150));
      expect(terminal.scrollLines).toHaveBeenCalled();
      cleanup();
    });

    it("isScrolling is true once the current gesture has scrolled", () => {
      const host = new EventTarget() as HTMLElement;
      const terminal = term(0, 200);
      const handle = installTerminalTouchScroll(host, terminal, {
        containGesture: true,
        pxPerLine: 10,
      });

      host.dispatchEvent(touchEvent("touchstart", 300));
      host.dispatchEvent(touchEvent("touchmove", 296));
      expect(handle.isScrolling()).toBe(false);
      host.dispatchEvent(touchEvent("touchmove", 280));
      expect(handle.isScrolling()).toBe(true);
      host.dispatchEvent(touchEvent("touchend", 280));

      host.dispatchEvent(touchEvent("touchstart", 280));
      expect(handle.isScrolling()).toBe(false);
      handle();
    });
  });
});
