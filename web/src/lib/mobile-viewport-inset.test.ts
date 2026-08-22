// @vitest-environment jsdom
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  MOBILE_BOTTOM_INSET_VAR,
  MOBILE_INSET_SAMPLE_MS,
  sampleViewportInset,
  trackMobileViewportInset,
} from "./mobile-viewport-inset";

const root = document.documentElement;

/** Fake `visualViewport` + `innerHeight` pair (layout px, visible px). */
function setViewport(layoutHeight: number, visibleHeight: number, offsetTop = 0) {
  Object.defineProperty(window, "innerHeight", {
    configurable: true,
    value: layoutHeight,
  });
  Object.defineProperty(window, "visualViewport", {
    configurable: true,
    value: {
      height: visibleHeight,
      offsetTop,
      addEventListener() {},
      removeEventListener() {},
    },
  });
}

function insetPx(): string {
  return root.style.getPropertyValue(MOBILE_BOTTOM_INSET_VAR);
}

beforeEach(() => {
  vi.useFakeTimers();
  root.style.removeProperty(MOBILE_BOTTOM_INSET_VAR);
  setViewport(800, 800);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  root.style.removeProperty(MOBILE_BOTTOM_INSET_VAR);
});

describe("sampleViewportInset", () => {
  it("returns null when visualViewport is unusable", () => {
    Object.defineProperty(window, "visualViewport", {
      configurable: true,
      value: null,
    });
    expect(sampleViewportInset()).toBeNull();
  });

  it("reads keyboard-sized regions as keyboard inset", () => {
    // 320px keyboard under an 800px layout.
    setViewport(800, 480);
    expect(sampleViewportInset()).toEqual({
      keyboardInset: 320,
      rawChrome: 0,
    });
  });

  it("reads bar-sized regions as the raw chrome candidate", () => {
    // 56px bottom bar.
    setViewport(800, 744);
    expect(sampleViewportInset()).toEqual({
      keyboardInset: 0,
      rawChrome: 56,
    });
  });
});

describe("trackMobileViewportInset", () => {
  it("is a no-op when visualViewport never provides a signal", () => {
    Object.defineProperty(window, "visualViewport", {
      configurable: true,
      value: null,
    });
    const stop = trackMobileViewportInset();
    vi.advanceTimersByTime(MOBILE_INSET_SAMPLE_MS * 4);
    expect(insetPx()).toBe("0px");
    stop();
  });

  it("applies a keyboard-sized inset immediately", () => {
    setViewport(800, 480); // 320px keyboard
    const stop = trackMobileViewportInset();
    expect(insetPx()).toBe("320px");
    stop();
    expect(insetPx()).toBe("0px");
  });

  it("applies a parked bar only after the stability run completes", () => {
    setViewport(800, 744); // 56px bottom bar, constant
    const stop = trackMobileViewportInset();
    // First (synchronous) sample starts the stability run.
    expect(insetPx()).toBe("0px");
    // Second consecutive equal sample confirms it.
    vi.advanceTimersByTime(MOBILE_INSET_SAMPLE_MS);
    expect(insetPx()).toBe("56px");
    // ...and it holds while the bar stays parked.
    vi.advanceTimersByTime(MOBILE_INSET_SAMPLE_MS * 5);
    expect(insetPx()).toBe("56px");
    stop();
  });

  it("never applies a wobbly (collapsing bar) value", () => {
    const stop = trackMobileViewportInset();
    // The bar expands/collapses with scroll — the value never settles,
    // so the stability run resets on every sample.
    const values = [748, 744, 749, 746, 744, 748, 744, 749];
    values.forEach((v) => {
      setViewport(800, v);
      vi.advanceTimersByTime(MOBILE_INSET_SAMPLE_MS);
      expect(insetPx()).toBe("0px");
    });
    stop();
  });

  it("drops a wobbly bar that then settles", () => {
    const stop = trackMobileViewportInset();
    const values = [748, 744, 749, 744, 744, 744];
    values.forEach((v) => {
      setViewport(800, v);
      vi.advanceTimersByTime(MOBILE_INSET_SAMPLE_MS);
    });
    expect(insetPx()).toBe("56px");
    stop();
  });

  it("lets a keyboard win over an unsettled bar in the same sample", () => {
    const stop = trackMobileViewportInset();
    // 320px keyboard with a 56px bar still visible below it: the raw
    // obscured region is 376px and the keyboard path (≥ 80px) accounts
    // for all of it — applied immediately.
    setViewport(800, 424);
    vi.advanceTimersByTime(MOBILE_INSET_SAMPLE_MS);
    expect(insetPx()).toBe("376px");
    // Keyboard dismissed: only the bar remains (56px) and it settles
    // into the inset once the stability run completes.
    setViewport(800, 744);
    vi.advanceTimersByTime(MOBILE_INSET_SAMPLE_MS * 2);
    expect(insetPx()).toBe("56px");
    stop();
  });

  it("clears the variable when stopped", () => {
    setViewport(800, 480);
    const stop = trackMobileViewportInset();
    expect(insetPx()).toBe("320px");
    stop();
    expect(insetPx()).toBe("0px");
  });

  it("is idempotent — a second start returns the same stop function", () => {
    const stop = trackMobileViewportInset();
    const stop2 = trackMobileViewportInset();
    expect(stop2).toBe(stop);
    stop();
    expect(insetPx()).toBe("0px");
    // After a stop the tracker can be started again.
    setViewport(800, 480);
    const stop3 = trackMobileViewportInset();
    expect(insetPx()).toBe("320px");
    stop3();
  });
});