import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { installTerminalLongPress, LONG_PRESS_MS } from "./terminal-long-press";

function touchEvent(type: string, x: number, y: number, fingers = 1): TouchEvent {
  const ev = new Event(type, { bubbles: true, cancelable: true }) as TouchEvent;
  const touch = { clientX: x, clientY: y } as Touch;
  const touches = type === "touchend" || type === "touchcancel" ? [] : Array(fingers).fill(touch);
  Object.defineProperty(ev, "touches", { value: touches });
  Object.defineProperty(ev, "changedTouches", { value: [touch] });
  return ev;
}

describe("installTerminalLongPress", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("fires after a still one-finger hold, without blocking the touch", () => {
    const host = new EventTarget() as HTMLElement;
    const onLongPress = vi.fn();
    const cleanup = installTerminalLongPress(host, onLongPress);

    const start = touchEvent("touchstart", 50, 50);
    host.dispatchEvent(start);
    const jitter = touchEvent("touchmove", 53, 52);
    host.dispatchEvent(jitter);
    vi.advanceTimersByTime(LONG_PRESS_MS);

    expect(onLongPress).toHaveBeenCalledTimes(1);
    expect(start.defaultPrevented).toBe(false);
    expect(jitter.defaultPrevented).toBe(false);
    cleanup();
  });

  it("is cancelled by a drag, a second finger, or lifting early", () => {
    const host = new EventTarget() as HTMLElement;
    const onLongPress = vi.fn();
    const cleanup = installTerminalLongPress(host, onLongPress);

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    host.dispatchEvent(touchEvent("touchmove", 50, 80));
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    host.dispatchEvent(touchEvent("touchmove", 50, 50, 2));
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    host.dispatchEvent(touchEvent("touchend", 50, 50));
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);

    expect(onLongPress).not.toHaveBeenCalled();
    cleanup();
  });

  it("claims a touch-originated contextmenu but leaves mouse right-click alone", () => {
    const host = new EventTarget() as HTMLElement;
    const onLongPress = vi.fn();
    const cleanup = installTerminalLongPress(host, onLongPress);

    const mouseMenu = new Event("contextmenu", { cancelable: true }) as MouseEvent;
    host.dispatchEvent(mouseMenu);
    expect(onLongPress).not.toHaveBeenCalled();
    expect(mouseMenu.defaultPrevented).toBe(false);

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    const touchMenu = new Event("contextmenu", { cancelable: true }) as MouseEvent;
    host.dispatchEvent(touchMenu);
    expect(onLongPress).toHaveBeenCalledTimes(1);
    expect(touchMenu.defaultPrevented).toBe(true);

    // The timer from that touchstart was cancelled, so no double fire.
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(onLongPress).toHaveBeenCalledTimes(1);
    cleanup();
  });

  it("removes every listener and pending timer on cleanup", () => {
    const host = new EventTarget() as HTMLElement;
    const onLongPress = vi.fn();
    const cleanup = installTerminalLongPress(host, onLongPress);

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    cleanup();
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    const menu = new Event("contextmenu", { cancelable: true }) as MouseEvent;
    host.dispatchEvent(menu);

    expect(onLongPress).not.toHaveBeenCalled();
    expect(menu.defaultPrevented).toBe(false);
  });

  it("a contextmenu after the finger drifted does not open the overlay", () => {
    const host = new EventTarget() as HTMLElement;
    const onLongPress = vi.fn();
    const cleanup = installTerminalLongPress(host, onLongPress);

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    host.dispatchEvent(touchEvent("touchmove", 50, 120));
    const menu = new Event("contextmenu", { cancelable: true }) as MouseEvent;
    host.dispatchEvent(menu);

    expect(onLongPress).not.toHaveBeenCalled();
    // Still swallowed so xterm doesn't treat it as a right-click.
    expect(menu.defaultPrevented).toBe(true);
    cleanup();
  });

  it("shouldTrigger can veto a still hold (e.g. it is part of a scroll)", () => {
    const host = new EventTarget() as HTMLElement;
    const onLongPress = vi.fn();
    let scrolling = true;
    const cleanup = installTerminalLongPress(host, onLongPress, {
      shouldTrigger: () => !scrolling,
    });

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).not.toHaveBeenCalled();
    host.dispatchEvent(touchEvent("touchend", 50, 50));

    scrolling = false;
    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    vi.advanceTimersByTime(LONG_PRESS_MS);
    expect(onLongPress).toHaveBeenCalledTimes(1);
    cleanup();
  });

  it("acceptTarget ignores holds on excluded targets", () => {
    const host = new EventTarget() as HTMLElement;
    const onLongPress = vi.fn();
    const cleanup = installTerminalLongPress(host, onLongPress, { acceptTarget: () => false });

    host.dispatchEvent(touchEvent("touchstart", 50, 50));
    vi.advanceTimersByTime(LONG_PRESS_MS * 2);
    expect(onLongPress).not.toHaveBeenCalled();
    cleanup();
  });
});
