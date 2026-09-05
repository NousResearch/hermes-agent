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

});
