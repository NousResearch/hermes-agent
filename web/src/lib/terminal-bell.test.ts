// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { resetBellContextForTests, ringBell } from "@/lib/terminal-bell";

class FakeGainNode {
  connect = vi.fn(() => this);
  gain = {
    exponentialRampToValueAtTime: vi.fn(),
    linearRampToValueAtTime: vi.fn(),
    setValueAtTime: vi.fn(),
  };
}

class FakeOscillatorNode {
  connect = vi.fn((node: unknown) => node);
  frequency = { value: 0 };
  start = vi.fn();
  stop = vi.fn();
  type = "";
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = [];

  currentTime = 7;
  destination = { isDestination: true };
  gains: FakeGainNode[] = [];
  oscillators: FakeOscillatorNode[] = [];
  resume = vi.fn(async () => {});
  state: AudioContextState = "suspended";

  constructor() {
    FakeAudioContext.instances.push(this);
  }

  createGain() {
    const gain = new FakeGainNode();
    this.gains.push(gain);
    return gain;
  }

  createOscillator() {
    const oscillator = new FakeOscillatorNode();
    this.oscillators.push(oscillator);
    return oscillator;
  }
}

function installAudioContext(ctor: unknown) {
  Object.defineProperty(window, "AudioContext", {
    configurable: true,
    value: ctor,
    writable: true,
  });
}

describe("terminal bell", () => {
  beforeEach(() => {
    FakeAudioContext.instances = [];
    resetBellContextForTests();
    // jsdom ships no WebAudio, so the default state of these tests is "no
    // AudioContext on window" unless a test installs one.
    delete (window as { AudioContext?: unknown }).AudioContext;
  });

  afterEach(() => {
    resetBellContextForTests();
    delete (window as { AudioContext?: unknown }).AudioContext;
  });

  it("stays silent instead of throwing when the browser has no WebAudio", () => {
    expect(() => ringBell()).not.toThrow();
    expect(FakeAudioContext.instances).toHaveLength(0);
  });

  it("plays two blips and resumes a suspended context", () => {
    installAudioContext(FakeAudioContext);

    ringBell();

    expect(FakeAudioContext.instances).toHaveLength(1);
    const context = FakeAudioContext.instances[0];
    expect(context.oscillators).toHaveLength(2);
    expect(context.resume).toHaveBeenCalledTimes(1);
    for (const oscillator of context.oscillators) {
      expect(oscillator.type).toBe("sine");
      expect(oscillator.start).toHaveBeenCalledTimes(1);
      expect(oscillator.stop).toHaveBeenCalledTimes(1);
      expect(oscillator.frequency.value).toBeGreaterThan(0);
    }
    // Regression guard from the 2026-10-16 field test: 0.09 peak gain was
    // audible "but only faintly" on the dashboard, so the chime must stay
    // loud enough to cut through without the user hunting for the tab.
    expect(context.gains).toHaveLength(2);
    for (const gain of context.gains) {
      const [peak] = gain.gain.linearRampToValueAtTime.mock.calls[0] ?? [];
      expect(peak).toBeGreaterThanOrEqual(0.25);
    }
  });

  it("reuses one context across rings so a chatty approval does not leak them", () => {
    installAudioContext(FakeAudioContext);

    ringBell();
    ringBell();

    expect(FakeAudioContext.instances).toHaveLength(1);
    expect(FakeAudioContext.instances[0].oscillators).toHaveLength(4);
  });

  it("swallows a constructor rejected by policy rather than breaking the terminal", () => {
    installAudioContext(
      class {
        constructor() {
          throw new Error("blocked by permissions policy");
        }
      },
    );

    expect(() => ringBell()).not.toThrow();
  });
});
