// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ChatVoiceControls,
} from "./ChatVoiceControls";
import { normalizeVoicePrompt, recognitionTranscript, sendVoicePrompt } from "@/lib/chat-voice";

const speechMocks = vi.hoisted(() => ({
  speak: vi.fn(async () => undefined),
  stop: vi.fn(),
}));

const feedMocks = vi.hoisted(() => {
  class FakeEventsFeed {
    static instances: FakeEventsFeed[] = [];
    handlers = new Map<string, Set<(event: { type: string; payload: Record<string, unknown> }) => void>>();
    closeHandlers = new Set<(code?: number) => void>();
    stateHandlers = new Set<(state: string) => void>();
    lastCloseCode: number | null | undefined = null;

    constructor() {
      FakeEventsFeed.instances.push(this);
    }

    async connect() {
      for (const handler of this.stateHandlers) handler("open");
    }

    close() {}

    on(type: string, handler: (event: { type: string; payload: Record<string, unknown> }) => void) {
      const handlers = this.handlers.get(type) ?? new Set();
      handlers.add(handler);
      this.handlers.set(type, handlers);
      return () => handlers.delete(handler);
    }

    onClose(handler: (code?: number) => void) {
      this.closeHandlers.add(handler);
      return () => this.closeHandlers.delete(handler);
    }

    onState(handler: (state: string) => void) {
      this.stateHandlers.add(handler);
      return () => this.stateHandlers.delete(handler);
    }

    emit(type: string, payload: Record<string, unknown> = {}) {
      for (const handler of this.handlers.get(type) ?? []) {
        handler({ type, payload });
      }
    }
  }
  return { FakeEventsFeed };
});

vi.mock("@/lib/eventsFeedClient", () => ({ EventsFeedClient: feedMocks.FakeEventsFeed }));
vi.mock("@/utils/jarvisSpeechUtils", () => ({
  speakWithNabra: speechMocks.speak,
  stopNabraAudio: speechMocks.stop,
}));

class FakeRecognition {
  static instances: FakeRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  onstart: (() => void) | null = null;
  onresult: ((event: { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0?: { transcript?: string } }> }) => void) | null = null;
  onerror: ((event: { error?: string }) => void) | null = null;
  onend: (() => void) | null = null;

  constructor() {
    FakeRecognition.instances.push(this);
  }

  start() {
    this.onstart?.();
  }

  abort() {}
}

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function button(label: string): HTMLButtonElement {
  const found = Array.from(container.querySelectorAll("button")).find((item) =>
    item.textContent?.includes(label),
  );
  if (!found) throw new Error(`button not found: ${label}`);
  return found;
}

describe("ChatVoiceControls", () => {
  beforeEach(() => {
    feedMocks.FakeEventsFeed.instances = [];
    FakeRecognition.instances = [];
    speechMocks.speak.mockClear();
    speechMocks.stop.mockClear();
    Object.defineProperty(window, "SpeechRecognition", {
      configurable: true,
      value: FakeRecognition,
    });
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("normalizes dictated prompts and separates final from interim text", () => {
    expect(normalizeVoicePrompt("  open\n the\t calendar  ")).toBe("open the calendar");
    expect(
      recognitionTranscript({
        resultIndex: 0,
        results: [
          { 0: { transcript: "run the task" }, isFinal: true },
          { 0: { transcript: "please" }, isFinal: false },
        ],
      }),
    ).toEqual({ final: "run the task", interim: "please" });

    const socket = { readyState: WebSocket.OPEN, send: vi.fn() };
    let sendReturn: (() => void) | undefined;
    expect(sendVoicePrompt(socket, " run\n it ", (callback) => { sendReturn = callback; }, () => true)).toBe(true);
    expect(socket.send).toHaveBeenCalledWith("run it");
    sendReturn?.();
    expect(socket.send).toHaveBeenLastCalledWith("\r");
  });

  it("submits final speech through the active Hermes chat via Send button or silence timeout", async () => {
    vi.useFakeTimers();
    const onSubmit = vi.fn(() => true);
    await act(async () => {
      root.render(
        <ChatVoiceControls
          channel="chat-1"
          connected
          foreground="#fff"
          onSubmit={onSubmit}
        />,
      );
    });

    await act(async () => {
      button("Mic").click();
      vi.advanceTimersByTime(10);
    });
    const recognition = FakeRecognition.instances.at(-1)!;
    expect(recognition.continuous).toBe(true);
    expect(recognition.lang).toBe("en-US");

    act(() => {
      recognition.onresult?.({
        resultIndex: 0,
        results: [{ 0: { transcript: "create a todo" }, isFinal: true }],
      });
    });

    // Instant send button is visible with draft
    expect(container.textContent).toContain("create a todo");
    const sendBtn = button("Send");
    act(() => {
      sendBtn.click();
    });

    expect(onSubmit).toHaveBeenCalledWith("create a todo");
    expect(container.textContent).toContain("Sent to Hermes");
    vi.useRealTimers();
  });

  it("accumulates multiple speech chunks continuously and auto-submits on silence timeout", async () => {
    vi.useFakeTimers();
    const onSubmit = vi.fn(() => true);
    await act(async () => {
      root.render(
        <ChatVoiceControls
          channel="chat-1"
          connected
          foreground="#fff"
          onSubmit={onSubmit}
        />,
      );
    });

    await act(async () => {
      button("Mic").click();
      vi.advanceTimersByTime(10);
    });
    const recognition = FakeRecognition.instances.at(-1)!;

    // Chunk 1
    act(() => {
      recognition.onresult?.({
        resultIndex: 0,
        results: [{ 0: { transcript: "please help me" }, isFinal: true }],
      });
    });
    expect(container.textContent).toContain("please help me");

    // Pause briefly (less than timeout, e.g. 500ms) - should NOT submit yet
    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(onSubmit).not.toHaveBeenCalled();

    // Chunk 2 (user continues speaking)
    act(() => {
      recognition.onresult?.({
        resultIndex: 1,
        results: [
          { 0: { transcript: "please help me" }, isFinal: true },
          { 0: { transcript: "write a python test" }, isFinal: true },
        ],
      });
    });
    expect(container.textContent).toContain("please help me write a python test");

    // Advance past silence timeout
    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(onSubmit).toHaveBeenCalledWith("please help me write a python test");
    expect(container.textContent).toContain("Sent to Hermes");
    vi.useRealTimers();
  });

  it("cycles language modes (EN -> عربي -> Auto) and re-starts recognition with the right language", async () => {
    vi.useFakeTimers();
    await act(async () => {
      root.render(
        <ChatVoiceControls
          channel="chat-1"
          connected
          foreground="#fff"
          onSubmit={() => true}
        />,
      );
    });

    // Default is English as requested
    expect(container.textContent).toContain("EN");

    // Start mic in English
    await act(async () => {
      button("Mic").click();
      vi.advanceTimersByTime(10);
    });
    let rec = FakeRecognition.instances.at(-1)!;
    expect(rec.lang).toBe("en-US");

    // Toggle to Arabic (عربي)
    await act(async () => {
      button("EN").click();
      vi.advanceTimersByTime(100);
    });
    expect(container.textContent).toContain("عربي");
    rec = FakeRecognition.instances.at(-1)!;
    expect(rec.lang).toBe("ar-EG");

    // Toggle to Auto
    await act(async () => {
      button("عربي").click();
      vi.advanceTimersByTime(100);
    });
    expect(container.textContent).toContain("Auto");

    // In Auto mode, speaking Arabic adapts the lang
    rec = FakeRecognition.instances.at(-1)!;
    act(() => {
      rec.onresult?.({
        resultIndex: 0,
        results: [{ 0: { transcript: "عايزك تساعدني" }, isFinal: true }],
      });
    });
    expect(container.textContent).toContain("عايزك تساعدني");

    vi.useRealTimers();
  });

  it("speaks streamed assistant sentences and flushes the final clause", async () => {
    await act(async () => {
      root.render(
        <ChatVoiceControls
          channel="chat-1"
          connected
          foreground="#fff"
          onSubmit={() => true}
        />,
      );
    });
    await act(async () => button("Voice off").click());
    const feed = feedMocks.FakeEventsFeed.instances[0];

    await act(async () => {
      feed.emit("message.start");
      feed.emit("message.delta", { text: "Task complete. Remaining detail" });
      await Promise.resolve();
    });
    expect(speechMocks.speak).toHaveBeenCalledWith("Task complete.", "jarvis", false);

    await act(async () => {
      feed.emit("message.complete", { text: "Task complete. Remaining detail" });
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(speechMocks.speak).toHaveBeenCalledWith("Remaining detail", "jarvis", false);
  });
});
