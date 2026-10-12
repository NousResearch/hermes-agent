import { afterEach, describe, expect, it, vi } from "vitest";

import {
  COMPOSING_ENTER_FALLBACK_MS,
  createComposingKeyGuard,
  resolveComposingKeydown,
} from "./pty-composing-key";

const keydown = (key: string, keyCode: number, isComposing = false) => ({
  type: "keydown",
  key,
  keyCode,
  isComposing,
  preventDefault: vi.fn(),
});

describe("resolveComposingKeydown", () => {
  it("leaves keys alone outside a composition and keyCode-229 placeholders inside one", () => {
    expect(resolveComposingKeydown(keydown(" ", 32), false).consume).toBe(false);
    expect(resolveComposingKeydown(keydown("Unidentified", 229, true), true).consume).toBe(false);
    expect(resolveComposingKeydown({ ...keydown("Enter", 13, true), type: "keyup" }, true).consume).toBe(false);
  });

  it("consumes a real key during a composition and defers only Enter", () => {
    expect(resolveComposingKeydown(keydown(" ", 32, true), false)).toEqual({ consume: true, deferEnter: false });
    expect(resolveComposingKeydown(keydown("Enter", 13), true)).toEqual({ consume: true, deferEnter: true });
  });
});

describe("createComposingKeyGuard", () => {
  afterEach(() => vi.useRealTimers());

  // SwiftKey autocorrect at the space bar: xterm must not finalize the
  // composition from the keydown (that sends the uncorrected word); the
  // space still reaches the textarea so compositionend carries it.
  it("tracks the composition and consumes a real space only while it is open", () => {
    const target = new EventTarget();
    const guard = createComposingKeyGuard(target, vi.fn());

    expect(guard.onKeydown(keydown(" ", 32))).toBe(false);
    target.dispatchEvent(new Event("compositionstart"));
    expect(guard.isComposing()).toBe(true);
    const space = keydown(" ", 32);
    expect(guard.onKeydown(space)).toBe(true);
    expect(space.preventDefault).not.toHaveBeenCalled();
    target.dispatchEvent(new Event("compositionend"));
    expect(guard.onKeydown(keydown(" ", 32))).toBe(false);
    guard.dispose();
  });

  it("replays Enter once, after the composition has committed", () => {
    vi.useFakeTimers();
    const sendEnter = vi.fn();
    const target = new EventTarget();
    const guard = createComposingKeyGuard(target, sendEnter);

    target.dispatchEvent(new Event("compositionstart"));
    const enter = keydown("Enter", 13);
    expect(guard.onKeydown(enter)).toBe(true);
    expect(enter.preventDefault).toHaveBeenCalledOnce();
    expect(sendEnter).not.toHaveBeenCalled();

    target.dispatchEvent(new Event("compositionend"));
    expect(sendEnter).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(sendEnter).toHaveBeenCalledOnce();
    guard.dispose();
  });

  it("sends a deferred Enter anyway when no compositionend follows", () => {
    vi.useFakeTimers();
    const sendEnter = vi.fn();
    const target = new EventTarget();
    const guard = createComposingKeyGuard(target, sendEnter);

    target.dispatchEvent(new Event("compositionstart"));
    guard.onKeydown(keydown("Enter", 13));
    vi.advanceTimersByTime(COMPOSING_ENTER_FALLBACK_MS - 1);
    expect(sendEnter).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(sendEnter).toHaveBeenCalledOnce();
    guard.dispose();
  });

  it("drops a pending Enter and stops listening on dispose", () => {
    vi.useFakeTimers();
    const sendEnter = vi.fn();
    const target = new EventTarget();
    const guard = createComposingKeyGuard(target, sendEnter);

    target.dispatchEvent(new Event("compositionstart"));
    guard.onKeydown(keydown("Enter", 13));
    guard.dispose();
    target.dispatchEvent(new Event("compositionend"));
    vi.runAllTimers();
    expect(sendEnter).not.toHaveBeenCalled();
  });
});
