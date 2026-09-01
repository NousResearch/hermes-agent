// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const gateway = vi.hoisted(() => {
  class MockGateway {
    stateHandler: ((state: string) => void) | null = null;
    handlers = new Map<string, (event: { type: string; session_id?: string; payload?: unknown }) => void>();
    requests: Array<{ method: string; params: Record<string, unknown> }> = [];
    snapshot: Record<string, unknown> | null = null;
    interrupt: Promise<void> | null = null;
    onState(handler: (state: string) => void) {
      this.stateHandler = handler;
      handler("idle");
      return () => { this.stateHandler = null; };
    }
    on(type: string, handler: (event: { type: string; session_id?: string; payload?: unknown }) => void) {
      this.handlers.set(type, handler);
      return () => this.handlers.delete(type);
    }
    async connect() { this.stateHandler?.("open"); }
    async request<T>(method: string, params: Record<string, unknown>) {
      this.requests.push({ method, params });
      if (method === "file.attach" && params.name === "fail.txt") throw new Error("upload failed");
      if (method === "prompt.submit" && params.text === "failed prompt" && this.requests.filter(({ method: requestMethod, params: requestParams }) => requestMethod === "prompt.submit" && requestParams.text === "failed prompt").length === 1) throw new Error("submit failed");
      if (method === "session.interrupt" && this.interrupt) await this.interrupt;
      if (method === "session.activate" || method === "session.resume") return { session_id: "runtime-1", messages: [{ id: 7, role: "user", text: "previous prompt" }, { id: 8, role: "assistant", content: "previous answer" }], ...this.snapshot } as T;
      if (method === "complete.slash") return { items: [{ display: "/help", text: "/help" }], replace_from: 0 } as T;
      if (method === "model.options") return { providers: [{ slug: "openai-codex", models: ["gpt-5.6-luna", "gpt-5.6-sol"] }, { slug: "openrouter", models: ["minimax/minimax-m3:free"] }] } as T;
      return (method === "session.create" ? { session_id: "session-1" } : { status: "streaming" }) as T;
    }
    close() {}
    emit(type: string, payload?: unknown, session_id = "session-1", seq?: number) {
      this.handlers.get(type)?.({ type, payload: seq === undefined ? payload : { ...(payload as object ?? {}), seq }, session_id });
    }
  }
  return { instance: null as InstanceType<typeof MockGateway> | null, MockGateway };
});

vi.mock("@/lib/gatewayClient", () => ({
  GatewayClient: class extends gateway.MockGateway {
    constructor() {
      super();
      gateway.instance = this;
    }
  },
}));
vi.mock("@/contexts/useProfileScope", () => ({ useProfileScope: () => ({ profile: "thai-profile" }) }));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      app: { openNavigation: "Open navigation" },
      sessions: { title: "Sessions" },
    },
  }),
}));
vi.mock("@/components/ChatSessionList", () => ({
  ChatSessionList: ({ onNewChat }: { onNewChat?: () => void }) => createElement("aside", { "data-testid": "session-list" },
    createElement("button", { type: "button", onClick: () => window.history.pushState({}, "", "/chat?resume=durable-2") }, "Existing session"),
    createElement("button", { type: "button", onClick: onNewChat }, "New chat")),
}));

import NativeChatPage, { shouldSubmitComposerKey } from "./NativeChatPage";
import {
  nativeChatModelChoices,
  nativeChatSessionCreateParams,
} from "@/lib/native-chat-routing";

describe("NativeChatPage", () => {
  let root: Root;
  let host: HTMLDivElement;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
  });

  it("keeps one native header and moves navigation controls into it", async () => {
    const onOpenNavigation = vi.fn();
    await act(async () => root.render(createElement(MemoryRouter, null,
      createElement(NativeChatPage, { onOpenNavigation }),
    )));

    expect(host.querySelectorAll("[data-slot='chat-header']")).toHaveLength(1);
    const openNavigation = host.querySelector<HTMLButtonElement>("button[aria-label='Open navigation']");
    expect(openNavigation).toBeTruthy();
    await act(async () => openNavigation?.click());
    expect(onOpenNavigation).toHaveBeenCalledTimes(1);

    const sessionsToggle = host.querySelector<HTMLButtonElement>("[data-session-navigator-toggle]");
    const navigator = host.querySelector<HTMLElement>("#native-chat-session-navigator");
    expect(sessionsToggle?.getAttribute("aria-expanded")).toBe("false");
    expect(navigator?.getAttribute("data-mobile-open")).toBe("false");
    expect(navigator?.classList.contains("hidden")).toBe(true);

    await act(async () => sessionsToggle?.click());
    expect(sessionsToggle?.getAttribute("aria-expanded")).toBe("true");
    expect(navigator?.getAttribute("data-mobile-open")).toBe("true");
    expect(navigator?.classList.contains("hidden")).toBe(false);
  });

  it("keeps Enter inside Thai IME composition and submits only after composition ends", () => {
    expect(shouldSubmitComposerKey("Enter", false, true)).toBe(false);
    expect(shouldSubmitComposerKey("Enter", true, false)).toBe(false);
    expect(shouldSubmitComposerKey("Enter", false, false)).toBe(true);
  });

  it("covers Thai combining marks and composer key transitions", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const thaiComposed = "สวัสดี ั";
    const setDraft = (value: string) => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, value);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    };

    // jsdom can exercise React's event wiring and exact Unicode values, but
    // does not emulate a browser/OS IME's native composition machinery.
    await act(async () => {
      setDraft("สั");
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter", isComposing: true }));
      expect(gateway.instance?.requests.filter(({ method }) => method === "prompt.submit")).toHaveLength(0);
      textarea.dispatchEvent(new CompositionEvent("compositionstart", { bubbles: true, data: "ส" }));
      setDraft("ส");
      textarea.dispatchEvent(new CompositionEvent("compositionupdate", { bubbles: true, data: "สั" }));
      setDraft("สั");
      const composingEnter = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter", isComposing: true });
      textarea.dispatchEvent(composingEnter);
      expect(gateway.instance?.requests.filter(({ method }) => method === "prompt.submit")).toHaveLength(0);
      textarea.dispatchEvent(new CompositionEvent("compositionend", { bubbles: true, data: thaiComposed }));
      setDraft(thaiComposed);
      const shiftedEnter = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter", shiftKey: true });
      textarea.dispatchEvent(shiftedEnter);
      expect(gateway.instance?.requests.filter(({ method }) => method === "prompt.submit")).toHaveLength(0);
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
    });
    expect(gateway.instance?.requests.filter(({ method }) => method === "prompt.submit")).toHaveLength(1);
    expect(gateway.instance?.requests.find(({ method }) => method === "prompt.submit")?.params.text).toBe(thaiComposed);
  });

  it("routes slash navigation only while the completion popover is visible", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "/");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const beforePopover = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "ArrowDown" });
    textarea.dispatchEvent(beforePopover);
    expect(beforePopover.defaultPrevented).toBe(false);

    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 100)); });
    expect(host.querySelector("[role='listbox']")).toBeTruthy();

    const whileVisible = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "ArrowDown" });
    textarea.dispatchEvent(whileVisible);
    expect(whileVisible.defaultPrevented).toBe(true);

    const tab = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Tab" });
    await act(async () => textarea.dispatchEvent(tab));
    expect(tab.defaultPrevented).toBe(true);
    expect(textarea.value).toBe("/help");

    const shiftedEnter = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter", shiftKey: true });
    textarea.dispatchEvent(shiftedEnter);
    expect(shiftedEnter.defaultPrevented).toBe(false);
    expect(gateway.instance?.requests.filter(({ method }) => method === "prompt.submit")).toHaveLength(0);
  });

  it("fills the native draft from an empty-state quick prompt", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const quickPrompt = host.querySelector<HTMLButtonElement>("[data-testid='quick-prompt']");
    expect(quickPrompt).toBeTruthy();
    const prompt = quickPrompt?.textContent ?? "";
    await act(async () => quickPrompt?.click());
    expect(host.querySelector<HTMLTextAreaElement>("textarea")?.value).toBe(prompt);
  });

  it("puts an assistant message into the native draft from Use as prompt", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "Use this answer" });
      gateway.instance?.emit("message.complete");
    });
    const useAsPrompt = host.querySelector<HTMLButtonElement>("button[aria-label='Use assistant message as prompt']");
    expect(useAsPrompt).toBeTruthy();
    await act(async () => useAsPrompt?.click());
    expect(host.querySelector<HTMLTextAreaElement>("textarea")?.value).toBe("Use this answer");
    expect(host.querySelector("[data-testid='message-action-feedback']")?.textContent).toContain("Draft filled");
  });

  it("preserves exact pasted Thai Unicode in the submitted prompt", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const pastedThai = "กำลังทดสอบ ั\u0e33";
    await act(async () => {
      const pasteEvent = new Event("paste", { bubbles: true, cancelable: true });
      Object.defineProperty(pasteEvent, "clipboardData", { value: { files: [], getData: () => pastedThai } });
      textarea.dispatchEvent(pasteEvent);
      // Text insertion is a browser default action; model it explicitly in
      // jsdom so this test checks the value passed through the composer.
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, pastedThai);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
    });
    expect(gateway.instance?.requests.find(({ method }) => method === "prompt.submit")?.params.text).toBe(pastedThai);
  });

  it("stages image and file attachments with the gateway payloads", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const input = host.querySelector<HTMLInputElement>("input[type=file]");
    expect(input).toBeTruthy();
    const image = new File(["image-bytes"], "photo.png", { type: "image/png" });
    const textFile = new File(["hello"], "notes.txt", { type: "text/plain" });
    await act(async () => {
      Object.defineProperty(input, "files", { value: [image, textFile] });
      input?.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 100)); });
    expect(gateway.instance?.requests.map(({ method }) => method)).toContain("image.attach_bytes");
    expect(gateway.instance?.requests.map(({ method }) => method)).toContain("file.attach");
    const imageRequest = gateway.instance?.requests.find(({ method }) => method === "image.attach_bytes");
    expect(imageRequest?.params).toMatchObject({ session_id: "session-1", filename: "photo.png" });
    expect(imageRequest?.params.content_base64).toBeTruthy();
    expect(gateway.instance?.requests.find(({ method }) => method === "file.attach")?.params).toMatchObject({
      session_id: "session-1", name: "notes.txt", path: "", data_url: expect.stringContaining("data:text/plain"),
    });
  });

  it("keeps attachment order, supports removal, and shows failed uploads with retry", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const input = host.querySelector<HTMLInputElement>("input[type=file]")!;
    const first = new File(["a"], "a.txt", { type: "text/plain" });
    const second = new File(["b"], "b.txt", { type: "text/plain" });
    const failed = new File(["x"], "fail.txt", { type: "text/plain" });
    await act(async () => {
      Object.defineProperty(input, "files", { value: [first, second, failed] });
      input.dispatchEvent(new Event("change", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    expect(host.textContent).toMatch(/a\.txt.*b\.txt/s);
    const remove = host.querySelector<HTMLButtonElement>("button[aria-label='Remove a.txt']");
    expect(remove).toBeTruthy();
    await act(async () => remove?.click());
    expect(host.textContent).not.toContain("a.txt");
    expect(host.textContent).toContain("upload failed");
    expect(host.querySelector("button[aria-label='Retry fail.txt']")).toBeTruthy();
  });

  it("preserves browser drop and paste attachment order", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const form = host.querySelector("form")!;
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const dropped = new File(["d"], "d.txt", { type: "text/plain" });
    const pasted = new File(["p"], "p.txt", { type: "text/plain" });
    await act(async () => {
      const dropEvent = new Event("drop", { bubbles: true, cancelable: true });
      Object.defineProperty(dropEvent, "dataTransfer", { value: { files: [dropped], items: [] } });
      form.dispatchEvent(dropEvent);
      const pasteEvent = new Event("paste", { bubbles: true, cancelable: true });
      Object.defineProperty(pasteEvent, "clipboardData", { value: { files: [pasted] } });
      textarea.dispatchEvent(pasteEvent);
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    expect(host.textContent).toMatch(/d\.txt.*p\.txt/s);
  });

  it("places tool activity in the transcript in event order", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("tool.start", { tool_id: "tool-1", name: "first" });
      gateway.instance?.emit("tool.complete", { tool_id: "tool-1", name: "first", result: { ok: true } });
      gateway.instance?.emit("tool.start", { tool_id: "tool-2", name: "second" });
    });
    const transcript = host.querySelector("[data-testid='native-chat-transcript']");
    const timeline = host.querySelector("[data-testid='tool-timeline']");
    expect(timeline?.parentElement).toBe(transcript);
    expect(Array.from(timeline?.querySelectorAll("[data-tool-id]") ?? []).map((item) => item.getAttribute("data-tool-id"))).toEqual(["tool-1", "tool-2"]);
    expect(timeline?.querySelector("[data-tool-id='tool-1']")?.getAttribute("data-tool-state")).toBe("complete");
    expect(timeline?.querySelector("[data-tool-id='tool-2']")?.getAttribute("data-tool-state")).toBe("running");
  });

  it("renders tool cards and sends exact approval and clarify response payloads", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("tool.start", { tool_id: "tool-1", name: "terminal", context: "running" });
      gateway.instance?.emit("tool.progress", { tool_id: "tool-1", progress: "halfway" });
      gateway.instance?.emit("tool.complete", { tool_id: "tool-1", name: "terminal", summary: "done" });
      gateway.instance?.emit("approval.request", { request_id: "approval-1", command: "rm file", choices: ["once", "deny"] });
    });
    expect(host.textContent).toContain("terminal");
    expect(host.textContent).toContain("done");
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    expect(gateway.instance?.requests.at(-1)).toEqual({ method: "approval.respond", params: { choice: "once", request_id: "approval-1", session_id: "session-1" } });
    await act(async () => gateway.instance?.emit("clarify.request", { request_id: "clarify-1", question: "Which?", choices: ["A"] }));
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='A']")?.click());
    expect(gateway.instance?.requests.at(-1)).toEqual({ method: "clarify.respond", params: { answer: "A", request_id: "clarify-1", session_id: "session-1" } });
  });

  it("stops a streaming session and deduplicates replayed deltas", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => { gateway.instance?.emit("message.start", undefined); gateway.instance?.emit("message.delta", { text: "one", seq: 2 }); gateway.instance?.emit("message.delta", { text: "one", seq: 2 }); });
    expect(host.textContent).toContain("one");
    expect(host.textContent).not.toContain("oneone");
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    expect(gateway.instance?.requests.at(-1)).toEqual({ method: "session.interrupt", params: { session_id: "session-1" } });
  });

  it("shows a stop-in-progress state and does not issue duplicate interrupts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start"));
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { stop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain("Stopping…");
    expect(stop.disabled).toBe(true);
    await act(async () => stop.click());
    expect(gateway.instance?.requests.filter(({ method }) => method === "session.interrupt")).toHaveLength(1);
    await act(async () => releaseInterrupt());
  });

  it("offers resend for a failed submit without duplicating the user message", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, "failed prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new Event("change", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });
    expect(host.textContent).toContain("submit failed");
    expect(host.querySelectorAll("article")).toHaveLength(1);
    const resend = host.querySelector<HTMLButtonElement>("button[aria-label='Retry send']")!;
    await act(async () => resend.click());
    expect(gateway.instance?.requests.filter(({ method }) => method === "prompt.submit")).toHaveLength(2);
    expect(host.querySelectorAll("article")).toHaveLength(1);
    expect(host.textContent).not.toContain("submit failed");
  });

  it("disables composer controls while disconnected", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    expect(host.querySelector<HTMLTextAreaElement>("textarea")?.disabled).toBe(true);
    expect(host.querySelector<HTMLButtonElement>("button[aria-label='Add attachment']")?.disabled).toBe(true);
    expect(host.querySelector<HTMLButtonElement>("button[type='submit']")?.disabled).toBe(true);
  });

  it("re-attaches the same runtime session after reconnect", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(gateway.instance?.requests.map(({ method }) => method)).toContain("session.activate");
    expect(gateway.instance?.requests.find(({ method }) => method === "session.activate")?.params).toEqual({ session_id: "session-1", omit_messages: false });
  });
  it("does not duplicate a live assistant transcript when reconnect snapshot catches up", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", undefined, "session-1", 1);
      gateway.instance?.emit("message.delta", { text: "recovered answer" }, "session-1", 2);
    });
    gateway.instance!.snapshot = { messages: [{ id: 9, role: "assistant", text: "recovered answer" }] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect((host.textContent?.match(/recovered answer/g) ?? []).length).toBe(1);
  });
  it("restores pending approval and clarification requests from a resume snapshot", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      running: true,
      status: "waiting",
      messages: [{ id: 7, role: "user", text: "previous prompt" }, { id: 8, role: "assistant", content: "previous answer" }],
      pending_approval: { request_id: "approval-resumed", command: "rm file", choices: ["once", "deny"] },
      pending_clarify: { request_id: "clarify-resumed", question: "Which?", choices: ["A"] },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain("rm file");
    expect(host.textContent).toContain("Which?");
    expect(host.textContent).toContain("Working");
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    expect(gateway.instance?.requests.at(-1)).toMatchObject({ method: "approval.respond", params: { request_id: "approval-resumed" } });
  });

  it("ignores duplicate tools and events from another session", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("tool.start", { tool_id: "tool-dup", name: "terminal" }, "session-1", 4);
      gateway.instance?.emit("tool.start", { tool_id: "tool-dup", name: "terminal" }, "session-1", 4);
      gateway.instance?.emit("tool.start", { tool_id: "wrong-session", name: "leak" }, "other-session", 5);
    });
    expect(host.textContent).toContain("terminal");
    expect(host.textContent).not.toContain("leak");
    expect(host.querySelectorAll("[data-tool-id='tool-dup']")).toHaveLength(1);
  });

  it("renders streamed assistant Markdown and fenced code", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    expect(gateway.instance?.requests[0]).toEqual({
      method: "session.create",
      params: { close_on_disconnect: true, source: "dashboard", profile: "thai-profile" },
    });
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "## Result\n\n```ts\nconst answer = 42;\n```" });
      gateway.instance?.emit("message.complete");
      gateway.instance?.emit("status.update", { text: "Ready" });
    });
    expect(host.textContent).toContain("Ready");
    expect(host.textContent).toContain("Connected");
    expect(host.querySelector("h2")?.textContent).toBe("Result");
    expect(host.querySelector("pre")?.textContent).toContain("const answer = 42;");
    expect(host.querySelector("[data-code-language]")?.textContent).toBe("ts");
  });

  it("keeps user messages as plain text with whitespace preserved", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const text = "<strong>not markup</strong>\n  indented";
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, text);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[type='submit']")?.click());
    const userArticle = Array.from(host.querySelectorAll("article")).find((article) => article.textContent?.includes("You"));
    expect(userArticle?.querySelector("strong")).toBeNull();
    expect(userArticle?.textContent).toContain(text);
    expect(userArticle?.className).toContain("whitespace-pre-wrap");
  });

  it("uses the bottom-follow threshold for transcript auto-scroll", async () => {
    const { shouldFollowTranscript } = await import("./NativeChatPage");
    expect(shouldFollowTranscript(0)).toBe(true);
    expect(shouldFollowTranscript(96)).toBe(true);
    expect(shouldFollowTranscript(97)).toBe(false);
  });

  it("shows a scroll-to-bottom affordance after the reader moves away from the latest message", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const transcript = host.querySelector<HTMLDivElement>("[data-testid='native-chat-transcript']")!;
    Object.defineProperty(transcript, "scrollHeight", { configurable: true, value: 1000 });
    Object.defineProperty(transcript, "clientHeight", { configurable: true, value: 100 });
    transcript.scrollTop = 0;
    await act(async () => transcript.dispatchEvent(new Event("scroll", { bubbles: true })));
    const scrollButton = host.querySelector<HTMLButtonElement>("button[aria-label='Scroll to latest message']");
    expect(scrollButton).toBeTruthy();

    await act(async () => scrollButton?.click());
    expect(transcript.scrollTop).toBe(1000);
    expect(host.querySelector("button[aria-label='Scroll to latest message']")).toBeNull();
  });

  it("resumes the durable URL session and displays its transcript snapshot", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(gateway.instance?.requests[0]).toMatchObject({ method: "session.activate", params: { session_id: "durable-1", profile: "thai-profile" } });
    expect(host.textContent).toContain("previous prompt");
    expect(host.textContent).toContain("previous answer");
  });

  it("clears resume and creates a fresh session from New chat without stale tool activity", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("tool.complete", { tool_id: "old-tool", name: "terminal", summary: "old result" }, "runtime-1"));
    expect(host.textContent).toContain("old result");
    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(gateway.instance?.requests.map(({ method }) => method)).toContain("session.create");
    expect(host.textContent).not.toContain("old result");
  });

  it("renders the session list in the native layout", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    expect(host.querySelector("[data-testid='session-list']")).toBeTruthy();
  });

  it("exposes stable full-height workspace slots and responsive semantics", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    expect(host.querySelector("[data-slot='native-chat-shell']")).toBeTruthy();
    expect(host.querySelector("[data-slot='chat-header']")).toBeTruthy();
    expect(host.querySelector("[data-slot='chat-body']")).toBeTruthy();
    expect(host.querySelector("[data-slot='session-navigator'][role='complementary']")).toBeTruthy();
    expect(host.querySelector("[data-slot='transcript-pane'][role='region']")).toBeTruthy();
    expect(host.querySelector("[data-testid='native-chat-transcript'][data-slot='transcript']")).toBeTruthy();
    expect(host.querySelector("[data-slot='chat-status'][role='status']")).toBeTruthy();
    expect(host.querySelector("[data-slot='chat-composer'][aria-label='Message composer']")).toBeTruthy();
  });

  it("renders Adaptive, catalog models, and all native reasoning levels", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.querySelector("#native-chat-model")?.textContent).toMatch(/Adaptive.*GPT Luna.*GPT Sol.*MiniMax M3 Free/s);
    expect(host.querySelector("#native-chat-reasoning")?.textContent).toMatch(/Auto.*Low.*Medium.*High.*Max/s);
  });

  it("builds Adaptive payloads without model/provider/reasoning overrides", () => {
    expect(nativeChatSessionCreateParams("thai-profile", { reasoning: "auto" })).toEqual({
      close_on_disconnect: true, source: "dashboard", profile: "thai-profile",
    });
  });

  it("builds explicit model and reasoning payloads", () => {
    const choices = nativeChatModelChoices({ providers: [{ slug: "openai-codex", models: ["gpt-5.6-luna"] }] });
    expect(nativeChatSessionCreateParams(undefined, { model: choices[0], reasoning: "high" })).toEqual({
      close_on_disconnect: true, source: "dashboard", model: "gpt-5.6-luna", provider: "openai-codex", reasoning_effort: "high",
    });
  });

  it("starts a fresh session when routing selection changes", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const model = host.querySelector<HTMLSelectElement>("#native-chat-model")!;
    await act(async () => {
      model.value = "openai-codex:gpt-5.6-luna";
      model.dispatchEvent(new Event("change", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(gateway.instance?.requests.filter(({ method }) => method === "session.create")).toHaveLength(2);
  });
});
