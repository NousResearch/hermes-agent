import { describe, expect, it } from "vitest";
import {
  decayMomentum,
  DEFAULT_SCROLL_GAIN,
  MOMENTUM_STOP_VELOCITY_PX_PER_MS,
  resolveScrollGain,
  TouchScrollAccumulator,
} from "./xterm-touch-scroll";

describe("TouchScrollAccumulator", () => {
  it("keeps sub-line travel instead of rounding it away", () => {
    const acc = new TouchScrollAccumulator(20);
    expect(acc.push(6)).toBe(0);
    expect(acc.push(6)).toBe(0);
    expect(acc.push(6)).toBe(0);
    // 18px of travel so far: still short of one 20px line.
    expect(acc.push(6)).toBe(1);
  });

  it("returns whole lines for a long drag and carries the remainder", () => {
    const acc = new TouchScrollAccumulator(10);
    expect(acc.push(95)).toBe(9);
    expect(acc.push(5)).toBe(1);
  });

  it("scrolls backwards for upward finger travel", () => {
    const acc = new TouchScrollAccumulator(20);
    expect(acc.push(-45)).toBe(-2);
    expect(acc.push(-15)).toBe(-1);
  });

  it("ignores zero and non-finite travel", () => {
    const acc = new TouchScrollAccumulator(20);
    expect(acc.push(0)).toBe(0);
    expect(acc.push(Number.NaN)).toBe(0);
    expect(acc.push(Number.POSITIVE_INFINITY)).toBe(0);
  });

  it("falls back to a 1px line height rather than dividing by zero", () => {
    const acc = new TouchScrollAccumulator(0);
    expect(acc.push(3)).toBe(3);
  });

  it("reset() drops the leftover remainder", () => {
    const acc = new TouchScrollAccumulator(20);
    acc.push(19);
    acc.reset();
    expect(acc.push(2)).toBe(0);
  });
});

describe("decayMomentum", () => {
  it("decays over time and reaches a stop threshold", () => {
    let v = 1;
    for (let i = 0; i < 200; i += 1) {
      v = decayMomentum(v, 16);
    }
    expect(Math.abs(v)).toBeLessThan(MOMENTUM_STOP_VELOCITY_PX_PER_MS);
  });

  it("leaves speed untouched when no time passed", () => {
    expect(decayMomentum(0.5, 0)).toBe(0.5);
  });

  it("glides for a noticeable stretch after a flick", () => {
    // Half of the release speed should still be there ~170 ms later, so a
    // flick keeps travelling instead of stopping dead under the finger.
    const after170ms = decayMomentum(1, 170);
    expect(after170ms).toBeGreaterThan(0.45);
    expect(after170ms).toBeLessThan(0.55);
  });
});

describe("resolveScrollGain", () => {
  it("defaults to a faster-than-finger gain", () => {
    expect(resolveScrollGain(undefined)).toBe(DEFAULT_SCROLL_GAIN);
    expect(DEFAULT_SCROLL_GAIN).toBeGreaterThan(1);
  });

  it("honours an explicit gain, including exact 1:1 tracking", () => {
    expect(resolveScrollGain(1)).toBe(1);
    expect(resolveScrollGain(2.25)).toBe(2.25);
  });

  it("rejects nonsense gains", () => {
    expect(resolveScrollGain(0)).toBe(DEFAULT_SCROLL_GAIN);
    expect(resolveScrollGain(-3)).toBe(DEFAULT_SCROLL_GAIN);
    expect(resolveScrollGain(Number.NaN)).toBe(DEFAULT_SCROLL_GAIN);
  });
});

describe("gain applied to travel", () => {
  it("scales finger travel before quantising to whole lines", () => {
    const acc = new TouchScrollAccumulator(18);
    // The tablet case measured at 18px rows: a 130px drag used to scroll 7
    // lines at 1:1; with the default gain it moves 13.
    expect(acc.push(130 * DEFAULT_SCROLL_GAIN)).toBe(13);
    // 234px = 13 lines exactly, so the next push starts from a clean slate.
    expect(acc.push(65 * DEFAULT_SCROLL_GAIN)).toBe(6);
  });
});
