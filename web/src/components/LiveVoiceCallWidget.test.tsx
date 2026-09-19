// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LiveVoiceCallWidget } from "./LiveVoiceCallWidget";

// Mock SpeechRecognition
class MockSpeechRecognition {
  continuous = false;
  interimResults = false;
  lang = "en-US";
  onstart: (() => void) | null = null;
  onresult: ((e: any) => void) | null = null;
  onerror: ((e: any) => void) | null = null;
  onend: (() => void) | null = null;

  start = vi.fn(() => {
    if (this.onstart) this.onstart();
  });
  stop = vi.fn(() => {
    if (this.onend) this.onend();
  });
  abort = vi.fn();
}

(globalThis as any).SpeechRecognition = MockSpeechRecognition;
(globalThis as any).webkitSpeechRecognition = MockSpeechRecognition;

// Mock Web Audio API
class MockAudioContext {
  state = "running";
  currentTime = 0;
  resume = vi.fn().mockResolvedValue(undefined);
  close = vi.fn().mockResolvedValue(undefined);
  createMediaStreamSource = vi.fn().mockReturnValue({
    connect: vi.fn(),
    disconnect: vi.fn(),
  });
  createBiquadFilter = vi.fn().mockReturnValue({
    type: "highpass",
    frequency: { setValueAtTime: vi.fn() },
    Q: { setValueAtTime: vi.fn() },
    connect: vi.fn(),
    disconnect: vi.fn(),
  });
  createDynamicsCompressor = vi.fn().mockReturnValue({
    threshold: { setValueAtTime: vi.fn() },
    knee: { setValueAtTime: vi.fn() },
    ratio: { setValueAtTime: vi.fn() },
    attack: { setValueAtTime: vi.fn() },
    release: { setValueAtTime: vi.fn() },
    connect: vi.fn(),
    disconnect: vi.fn(),
  });
  createAnalyser = vi.fn().mockReturnValue({
    fftSize: 256,
    frequencyBinCount: 128,
    smoothingTimeConstant: 0.8,
    getByteFrequencyData: vi.fn((arr: Uint8Array) => arr.fill(10)),
    connect: vi.fn(),
    disconnect: vi.fn(),
  });
  createMediaStreamDestination = vi.fn().mockReturnValue({
    stream: {},
    disconnect: vi.fn(),
  });
}

(globalThis as any).AudioContext = MockAudioContext;
(globalThis as any).webkitAudioContext = MockAudioContext;

// Mock getUserMedia
if (!navigator.mediaDevices) {
  (navigator as any).mediaDevices = {};
}
navigator.mediaDevices.getUserMedia = vi.fn().mockResolvedValue({
  getTracks: () => [{ stop: vi.fn(), enabled: true }],
  getAudioTracks: () => [{ stop: vi.fn(), enabled: true }],
});

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("LiveVoiceCallWidget", () => {
  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });

  it("renders with DSP clean audio indicator and language tabs", async () => {
    await act(async () => {
      root.render(<LiveVoiceCallWidget />);
    });

    expect(container.textContent).toContain("DSP Voice Filter: Active");
    expect(container.textContent).toContain("عربي (EG)");
    expect(container.textContent).toContain("English");
    expect(container.textContent).toContain("Auto");
    expect(container.textContent).toContain("Start Call");
  });

  it("switches language to English and updates active styling", async () => {
    await act(async () => {
      root.render(<LiveVoiceCallWidget />);
    });

    const englishBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "English"
    );
    expect(englishBtn).toBeDefined();

    await act(async () => {
      englishBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(englishBtn?.className).toContain("bg-[#00f0ff]/20");
    expect(container.textContent).toContain("Language switched to English (US)");
  });

  it("switches language to Arabic (EG) cleanly", async () => {
    await act(async () => {
      root.render(<LiveVoiceCallWidget />);
    });

    const arabicBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "عربي (EG)"
    );
    expect(arabicBtn).toBeDefined();

    await act(async () => {
      arabicBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(arabicBtn?.className).toContain("bg-[#00f0ff]/20");
    expect(container.textContent).toContain("Language switched to Arabic - مصرية");
  });

  it("initiates live call and activates microphone with DSP audio pipeline", async () => {
    await act(async () => {
      root.render(<LiveVoiceCallWidget />);
    });

    const startCallBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("Start Call")
    );
    expect(startCallBtn).toBeDefined();

    await act(async () => {
      startCallBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalled();
    expect(container.textContent).toContain("End Call");
  });

  it("renders audio latency mode controls and toggles between Instant and Studio Voice", async () => {
    await act(async () => {
      root.render(<LiveVoiceCallWidget />);
    });

    expect(container.textContent).toContain("Instant Speech");
    expect(container.textContent).toContain("Studio Voice");
    expect(container.textContent).toContain("TTS: Instant (<1s)");

    const studioBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.includes("Studio Voice")
    );
    expect(studioBtn).toBeDefined();

    await act(async () => {
      studioBtn?.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(studioBtn?.className).toContain("bg-[#00f0ff]/20");
    expect(container.textContent).toContain("TTS: Pipelined");
  });
});

