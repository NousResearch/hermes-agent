// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ChatVoiceControl } from "./ChatVoiceControl";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

interface FakeResult { 0: { transcript: string }; isFinal: boolean }

class FakeRecognition {
  static instances: FakeRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  onresult: ((event: { resultIndex: number; results: FakeResult[] }) => void) | null = null;
  onend: (() => void) | null = null;
  onerror: ((event: { error: string }) => void) | null = null;
  start = vi.fn();
  stop = vi.fn();
  abort = vi.fn();
  constructor() { FakeRecognition.instances.push(this); }
  result(resultIndex: number, ...rows: Array<[string, boolean]>) {
    this.onresult?.({ resultIndex, results: rows.map(([transcript, isFinal]) => ({ 0: { transcript }, isFinal })) });
  }
}

class FakeNativeBridge {
  messages: string[] = [];
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  postMessage(message: string) { this.messages.push(message); }
  emit(event: string, text?: string) {
    this.onmessage?.(new MessageEvent("message", {
      data: JSON.stringify({ version: 1, event, ...(text === undefined ? {} : { text }) }),
    }));
  }
}

describe("ChatVoiceControl browser speech input", () => {
  let host: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;

  beforeEach(() => {
    vi.useFakeTimers();
    FakeRecognition.instances = [];
    vi.stubGlobal("SpeechRecognition", FakeRecognition);
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  async function render(submit = vi.fn(), connected = true) {
    await act(async () => root.render(<ChatVoiceControl connected={connected} submit={submit} />));
    return { submit, surface: host.querySelector('[aria-label="Browser voice input"]') as HTMLButtonElement };
  }

  it("accumulates final results while replacing interim text without submitting", async () => {
    const { submit, surface } = await render();
    await act(async () => surface.click());
    const recognition = FakeRecognition.instances[0];
    expect(recognition.continuous).toBe(true);
    expect(recognition.interimResults).toBe(true);
    await act(async () => recognition.result(0, ["Call the ", true], ["cr", false]));
    await act(async () => recognition.result(1, ["ignored", true], ["crew", false]));
    expect(host.textContent).toContain("Call the crew");
    expect(host.textContent).not.toContain("ignored");
    expect(submit).not.toHaveBeenCalled();
  });

  it("uses the browser locale for speech recognition", async () => {
    vi.stubGlobal("navigator", { language: "fr-CA" });
    const { surface } = await render();

    await act(async () => surface.click());

    expect(FakeRecognition.instances[0].lang).toBe("fr-CA");
  });

  it("restarts Android premature onend with bounded exponential backoff and no submit", async () => {
    const { submit, surface } = await render();
    await act(async () => surface.click());
    await act(async () => FakeRecognition.instances[0].result(0, ["Keep going", true]));
    await act(async () => FakeRecognition.instances[0].onend?.());
    await act(async () => vi.advanceTimersByTime(249));
    expect(FakeRecognition.instances).toHaveLength(1);
    await act(async () => vi.advanceTimersByTime(1));
    expect(FakeRecognition.instances).toHaveLength(2);
    await act(async () => FakeRecognition.instances[1].onend?.());
    await act(async () => vi.advanceTimersByTime(499));
    expect(FakeRecognition.instances).toHaveLength(2);
    await act(async () => vi.advanceTimersByTime(1));
    expect(FakeRecognition.instances).toHaveLength(3);
    expect(host.textContent).toContain("Keep going");
    expect(submit).not.toHaveBeenCalled();
  });

  it("stops retrying after the bounded restart budget", async () => {
    const { surface } = await render();
    await act(async () => surface.click());
    for (let index = 0; index < 6; index += 1) {
      await act(async () => FakeRecognition.instances.at(-1)?.onend?.());
      await act(async () => vi.runOnlyPendingTimers());
    }
    await act(async () => FakeRecognition.instances.at(-1)?.onend?.());
    await act(async () => vi.runAllTimers());
    expect(FakeRecognition.instances).toHaveLength(7);
    expect(host.textContent).toContain("kept stopping");
  });

  it("does not restart after a fatal permission error", async () => {
    const { surface } = await render();
    await act(async () => surface.click());
    const recognition = FakeRecognition.instances[0];
    await act(async () => recognition.onerror?.({ error: "not-allowed" }));
    await act(async () => recognition.onend?.());
    await act(async () => vi.runAllTimers());
    expect(FakeRecognition.instances).toHaveLength(1);
    expect(host.textContent).toContain("permission was denied");
  });

  it("submits one normalized transcript and aborts recognition", async () => {
    const { submit, surface } = await render();
    await act(async () => surface.click());
    await act(async () => FakeRecognition.instances[0].result(0, ["Call   the ", true], [" crew", false]));
    await act(async () => surface.click());
    await act(async () => surface.click());
    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit).toHaveBeenCalledWith("Call the crew");
    expect(FakeRecognition.instances[0].abort).toHaveBeenCalledOnce();
  });

  it("does not submit empty audio and remains listening", async () => {
    const { submit, surface } = await render();
    await act(async () => surface.click());
    await act(async () => surface.click());
    expect(submit).not.toHaveBeenCalled();
    expect(host.textContent).toContain("No speech heard");
    expect(host.textContent).toContain("TAP ANYWHERE TO SEND");
  });

  it("cancel aborts capture, clears text, and prevents restart", async () => {
    const { submit, surface } = await render();
    await act(async () => surface.click());
    const recognition = FakeRecognition.instances[0];
    await act(async () => recognition.result(0, ["do not send", true]));
    const cancel = host.querySelector('[aria-label="Cancel browser voice input"]') as HTMLButtonElement;
    await act(async () => cancel.click());
    await act(async () => recognition.onend?.());
    await act(async () => vi.runAllTimers());
    expect(recognition.abort).toHaveBeenCalledOnce();
    expect(submit).not.toHaveBeenCalled();
    expect(host.textContent).not.toContain("do not send");
  });

  it("keeps the terminal text-entry fallback available", async () => {
    const { surface } = await render();
    expect(surface.textContent).toContain("Gboard typed fallback remains below");
  });

  it("suppresses the listening pulse when reduced motion is requested", async () => {
    const { surface } = await render();
    await act(async () => surface.click());
    const mic = surface.querySelector("svg");
    expect(mic?.classList.contains("animate-pulse")).toBe(true);
    expect(mic?.classList.contains("motion-reduce:animate-none")).toBe(true);
  });

  it("refuses to start while chat is disconnected", async () => {
    const { surface } = await render(vi.fn(), false);
    await act(async () => surface.click());
    expect(FakeRecognition.instances).toHaveLength(0);
    expect(host.textContent).toContain("not connected");
  });

  it("checks the native bridge and falls back when on-device recognition is unavailable", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { surface } = await render();

    await act(async () => surface.click());
    expect(bridge.messages).toEqual([JSON.stringify({ version: 1, command: "check" })]);
    await act(async () => bridge.emit("availability", "on-device-unavailable;fallback-choice-required"));

    expect(FakeRecognition.instances).toHaveLength(1);
    expect(FakeRecognition.instances[0].start).toHaveBeenCalledOnce();
  });

  it("does not trust ready or final events before confirmed native availability", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { submit, surface } = await render();

    await act(async () => surface.click());
    await act(async () => bridge.emit("ready"));
    await act(async () => bridge.emit("final", "must not send"));
    await act(async () => vi.advanceTimersByTime(1_000));

    expect(submit).not.toHaveBeenCalled();
    expect(FakeRecognition.instances).toHaveLength(1);
  });

  it("logs and ignores a malformed native bridge payload", async () => {
    const bridge = new FakeNativeBridge();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { surface } = await render();

    await act(async () => surface.click());
    await act(async () => bridge.onmessage?.(new MessageEvent("message", { data: "not-json" })));

    expect(warn).toHaveBeenCalledWith("[voice] Ignoring malformed native bridge payload");
    expect(FakeRecognition.instances).toHaveLength(0);
    await act(async () => bridge.emit("availability", "on-device"));
    expect(bridge.messages.at(-1)).toBe(JSON.stringify({ version: 1, command: "start" }));
  });

  it("uses confirmed native partials for display and submits a final exactly once", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { submit, surface } = await render();

    await act(async () => surface.click());
    await act(async () => bridge.emit("availability", "on-device"));
    expect(bridge.messages.at(-1)).toBe(JSON.stringify({ version: 1, command: "start" }));
    await act(async () => bridge.emit("ready"));
    await act(async () => bridge.emit("partial", "Call the cr"));
    expect(host.textContent).toContain("Call the cr");
    expect(submit).not.toHaveBeenCalled();
    const staleHandler = bridge.onmessage;
    await act(async () => bridge.emit("final", "Call   the crew"));
    await act(async () => staleHandler?.(new MessageEvent("message", {
      data: JSON.stringify({ version: 1, event: "final", text: "duplicate" }),
    })));

    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit).toHaveBeenCalledWith("Call the crew");
  });

  it("falls back after a bounded native handshake timeout", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { surface } = await render();
    await act(async () => surface.click());
    await act(async () => vi.advanceTimersByTime(999));
    expect(FakeRecognition.instances).toHaveLength(0);
    await act(async () => vi.advanceTimersByTime(1));
    expect(FakeRecognition.instances).toHaveLength(1);
    expect(bridge.messages.at(-1)).toBe(JSON.stringify({ version: 1, command: "cancel" }));
  });

  it("rejects every retained native callback after timeout fallback", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { submit, surface } = await render();
    await act(async () => surface.click());
    const staleHandler = bridge.onmessage;

    await act(async () => vi.advanceTimersByTime(1_000));
    expect(FakeRecognition.instances).toHaveLength(1);
    const displayAfterFallback = host.textContent;

    for (const [event, text] of [
      ["availability", "on-device"],
      ["ready", undefined],
      ["partial", "stale partial"],
      ["final", "stale final"],
      ["error", "stale error"],
    ] as const) {
      await act(async () => staleHandler?.(new MessageEvent("message", {
        data: JSON.stringify({ version: 1, event, ...(text === undefined ? {} : { text }) }),
      })));
    }

    expect(host.textContent).toBe(displayAfterFallback);
    expect(submit).not.toHaveBeenCalled();
    expect(FakeRecognition.instances).toHaveLength(1);
    expect(FakeRecognition.instances[0].start).toHaveBeenCalledOnce();
  });

  it.each(["ended", "error"])("falls back when native emits %s while listening", async (event) => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { surface } = await render();
    await act(async () => surface.click());
    await act(async () => bridge.emit("availability", "on-device"));
    await act(async () => bridge.emit("listening"));
    await act(async () => bridge.emit(event, "native-ended"));
    await act(async () => vi.advanceTimersByTime(500));
    expect(FakeRecognition.instances).toHaveLength(1);
    expect(FakeRecognition.instances[0].start).toHaveBeenCalledOnce();
  });

  it("accepts the concrete Android consumer final that follows speech-ended", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { submit, surface } = await render();
    await act(async () => surface.click());
    await act(async () => bridge.emit("availability", "on-device"));
    await act(async () => bridge.emit("ready"));
    await act(async () => bridge.emit("ended", "speech-ended"));
    await act(async () => bridge.emit("final", "Android final"));
    await act(async () => vi.advanceTimersByTime(500));
    expect(submit).toHaveBeenCalledOnce();
    expect(submit).toHaveBeenCalledWith("Android final");
    expect(FakeRecognition.instances).toHaveLength(0);
  });

  it("falls back if native stop never produces a final or ended event", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { surface } = await render();
    await act(async () => surface.click());
    await act(async () => bridge.emit("availability", "on-device"));
    await act(async () => bridge.emit("ready"));
    await act(async () => surface.click());
    expect(bridge.messages.at(-1)).toBe(JSON.stringify({ version: 1, command: "stop" }));
    await act(async () => vi.advanceTimersByTime(1_000));
    expect(FakeRecognition.instances).toHaveLength(1);
  });

  it("cancels the native bridge and detaches its event handler on unmount", async () => {
    const bridge = new FakeNativeBridge();
    Object.defineProperty(window, "zer0Voice", { configurable: true, value: bridge });
    const { surface } = await render();
    await act(async () => surface.click());
    await act(async () => root.unmount());
    expect(bridge.messages.at(-1)).toBe(JSON.stringify({ version: 1, command: "cancel" }));
    expect(bridge.onmessage).toBeNull();
    root = createRoot(host);
  });

});
