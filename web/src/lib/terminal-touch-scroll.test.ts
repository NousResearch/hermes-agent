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
});
