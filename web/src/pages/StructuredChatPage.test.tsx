// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import StructuredChatPage, { type StructuredGateway } from "./StructuredChatPage";

let root: Root | undefined;
let container: HTMLDivElement | undefined;

async function render(
  client: StructuredGateway,
  followEvents?: (
    sessionIds: string[],
    onEvent: (event: { session_id?: string; seq?: unknown; type: string; payload?: unknown }) => void,
  ) => () => void,
) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <StructuredChatPage
        clientFactory={() => client}
        search="?resume=durable-1&profile=worker"
        followEvents={followEvents}
      />,
    );
  });
}

async function settle() {
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
}

beforeEach(() => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = undefined;
  container = undefined;
});

describe("StructuredChatPage", () => {
  it("resumes the exact session, loads history, and clears native composer only after submit acknowledgement", async () => {
    let eventHandler: ((event: Parameters<NonNullable<StructuredGateway["onAny"]>>[0] extends (event: infer E) => void ? E : never) => void) | undefined;
    let acceptSubmit: (() => void) | undefined;
    const calls: Array<[string, Record<string, unknown> | undefined]> = [];
    const client: StructuredGateway = {
      connect: vi.fn(async () => undefined),
      close: vi.fn(),
      onAny: vi.fn((handler) => { eventHandler = handler; return () => undefined; }),
      onState: vi.fn((handler) => { handler("open"); return () => undefined; }),
      request: vi.fn(async (method, params) => {
        calls.push([method, params]);
        if (method === "session.resume") {
          return {
            session_id: "runtime-1",
            running: false,
            ownership_epoch: 3,
            read_only: false,
            messages: [
              { role: "assistant", text: "Bestehend" },
              { role: "assistant", text: "Aktueller PTY-Schwanz" },
            ],
          };
        }
        if (method === "session.history") return { messages: [{ role: "assistant", text: "Bestehend" }] };
        if (method === "prompt.submit") return new Promise((resolve) => { acceptSubmit = () => resolve({ status: "streaming" }); });
        return {};
      }),
    };

    await render(client);
    await settle();

    expect(calls[0]?.[0]).toBe("session.resume");
    expect(calls[0]?.[1]).toEqual(expect.objectContaining({
      session_id: "durable-1",
      profile: "worker",
      owner_id: expect.any(String),
    }));
    expect(calls.some(([method]) => method === "session.history")).toBe(false);
    expect(container?.textContent).toContain("Bestehend");
    expect(container?.textContent).toContain("Aktueller PTY-Schwanz");

    const composer = container?.querySelector("textarea") as HTMLTextAreaElement;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(composer, "Diktierter Text");
      composer.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: "Diktierter Text" }));
    });
    await act(async () => composer.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })));
    expect(composer.value).toBe("Diktierter Text");
    expect(calls.at(-1)?.[0]).toBe("prompt.submit");
    expect(calls.at(-1)?.[1]).toEqual(expect.objectContaining({
      session_id: "runtime-1",
      text: "Diktierter Text",
      owner_id: expect.any(String),
      ownership_epoch: 3,
    }));

    await act(async () => acceptSubmit?.());
    expect(composer.value).toBe("");
    expect(container?.textContent).toContain("Diktierter Text");
    expect(eventHandler).toBeTypeOf("function");
  });

  it("keeps rejected text and enters an explicit read-only state when another runtime owns the session", async () => {
    const client: StructuredGateway = {
      connect: async () => undefined,
      close: vi.fn(),
      onAny: vi.fn(() => () => undefined),
      onState: vi.fn((handler) => { handler("open"); return () => undefined; }),
      request: vi.fn(async (method) => {
        if (method === "session.resume") return { session_id: "runtime-1", running: false };
        if (method === "session.history") return { messages: [] };
        if (method === "prompt.submit") throw new Error("SESSION_NOT_OWNED");
        return {};
      }),
    };
    await render(client);
    await settle();
    const composer = container?.querySelector("textarea") as HTMLTextAreaElement;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(composer, "Nicht verlieren");
      composer.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: "Nicht verlieren" }));
    });
    await act(async () => composer.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })));
    await settle();

    expect(composer.value).toBe("Nicht verlieren");
    expect(composer.disabled).toBe(true);
    expect(container?.textContent).toContain("Eine andere Laufzeit besitzt diese Session");
  });

  it("answers an approval through its typed gateway operation and stable request id", async () => {
    let emit: ((event: { session_id?: string; seq?: unknown; type: string; payload?: unknown }) => void) | undefined;
    const calls: Array<[string, Record<string, unknown> | undefined]> = [];
    const client: StructuredGateway = {
      connect: async () => undefined,
      close: vi.fn(),
      onAny: vi.fn((handler) => { emit = handler; return () => undefined; }),
      onState: vi.fn((handler) => { handler("open"); return () => undefined; }),
      request: vi.fn(async (method, params) => {
        calls.push([method, params]);
        if (method === "session.resume") return { session_id: "runtime-1", running: false };
        if (method === "session.history") return { messages: [] };
        return { accepted: true };
      }),
    };
    await render(client);
    await settle();
    await act(async () => emit?.({
      session_id: "runtime-1",
      seq: 1,
      type: "approval.request",
      payload: { request_id: "approval-7", message: "Befehl erlauben?" },
    }));

    const approve = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "Erlauben");
    expect(approve).toBeDefined();
    await act(async () => approve?.click());
    await settle();

    expect(calls.at(-1)).toEqual(["approval.respond", {
      session_id: "runtime-1",
      request_id: "approval-7",
      choice: "once",
      owner_id: expect.any(String),
      ownership_epoch: 0,
    }]);
    expect(container?.querySelector('[data-request-id="approval-7"]')?.textContent).toContain("beantwortet");
  });

  it("appends live TUI output that arrives on the session event feed", async () => {
    let emitFollow: ((event: { session_id?: string; seq?: unknown; type: string; payload?: unknown }) => void) | undefined;
    const followed: string[][] = [];
    const client: StructuredGateway = {
      connect: async () => undefined,
      close: vi.fn(),
      onAny: vi.fn(() => () => undefined),
      onState: vi.fn((handler) => { handler("open"); return () => undefined; }),
      request: vi.fn(async (method) => {
        if (method === "session.resume") {
          return {
            session_id: "runtime-1",
            running: true,
            read_only: true,
            messages: [{ role: "assistant", text: "Bisher" }],
          };
        }
        return {};
      }),
    };
    await render(client, (sessionIds, onEvent) => {
      followed.push(sessionIds);
      emitFollow = onEvent;
      return () => undefined;
    });
    await settle();
    expect(container?.textContent).toContain("Bisher");
    await act(async () => emitFollow?.({
      session_id: "durable-1",
      seq: 4,
      type: "message.delta",
      payload: { text: "frisch aus der TUI" },
    }));
    expect(container?.textContent).toContain("frisch aus der TUI");
    expect(followed.some((ids) => ids.includes("durable-1") && ids.includes("runtime-1"))).toBe(true);
  });

  it("picks up later TUI turns from session.history", async () => {
    vi.useFakeTimers();
    try {
      const client: StructuredGateway = {
        connect: async () => undefined,
        close: vi.fn(),
        onAny: vi.fn(() => () => undefined),
        onState: vi.fn((handler) => { handler("open"); return () => undefined; }),
        request: vi.fn(async (method) => {
          if (method === "session.resume") {
            return {
              session_id: "runtime-1",
              running: true,
              read_only: true,
              messages: [{ role: "assistant", text: "Alt" }],
            };
          }
          if (method === "session.history") {
            return {
              messages: [
                { role: "assistant", text: "Alt" },
                { role: "assistant", text: "TUI weiter" },
              ],
            };
          }
          return {};
        }),
      };
      await render(client, () => () => undefined);
      await settle();
      expect(container?.textContent).toContain("Alt");
      expect(container?.textContent).not.toContain("TUI weiter");
      await act(async () => { await vi.advanceTimersByTimeAsync(2100); });
      await settle();
      expect(container?.textContent).toContain("TUI weiter");
    } finally {
      vi.useRealTimers();
    }
  });

  it("takeover after read-only resume admits later submits with the new epoch", async () => {
    const calls: Array<[string, Record<string, unknown> | undefined]> = [];
    const client: StructuredGateway = {
      connect: async () => undefined,
      close: vi.fn(),
      onAny: vi.fn(() => () => undefined),
      onState: vi.fn((handler) => { handler("open"); return () => undefined; }),
      request: vi.fn(async (method, params) => {
        calls.push([method, params]);
        if (method === "session.resume") {
          return { session_id: "runtime-1", running: false, read_only: true, ownership_epoch: 4 };
        }
        if (method === "session.history") return { messages: [] };
        if (method === "session.takeover") return { session_id: "runtime-1", ownership_epoch: 5, read_only: false };
        return {};
      }),
    };
    await render(client);
    await settle();
    const composer = container?.querySelector("textarea") as HTMLTextAreaElement;
    expect(composer.disabled).toBe(true);
    const takeover = Array.from(container?.querySelectorAll("button") ?? []).find((button) => button.textContent === "Session übernehmen");
    expect(takeover).toBeDefined();
    await act(async () => takeover?.click());
    await settle();
    expect(calls.some(([method, params]) => method === "session.takeover" && params?.confirmed === true)).toBe(true);
    expect(composer.disabled).toBe(false);
  });

  it("keeps a native iPhone composer and opens PTY only with an explicit diagnostic bypass", async () => {
    const client: StructuredGateway = {
      connect: async () => undefined,
      close: vi.fn(),
      onAny: vi.fn(() => () => undefined),
      onState: vi.fn((handler) => { handler("open"); return () => undefined; }),
      request: vi.fn(async (method) => {
        if (method === "session.resume") {
          return { session_id: "runtime-1", running: false, read_only: false, ownership_epoch: 1 };
        }
        return {};
      }),
    };
    await render(client);
    await settle();
    const composer = container?.querySelector("#structured-chat-composer") as HTMLTextAreaElement;
    expect(composer.getAttribute("autocapitalize")).toBe("sentences");
    expect(composer.getAttribute("inputmode")).toBe("text");
    expect(composer.getAttribute("spellcheck")).not.toBe("false");
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set?.call(composer, "Hallo Welt");
      composer.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: "Hallo Welt" }));
      composer.setSelectionRange(5, 5);
      composer.dispatchEvent(new Event("selectionchange", { bubbles: true }));
    });
    expect(composer.selectionStart).toBe(5);
    expect(composer.selectionEnd).toBe(5);
    const pty = container?.querySelector("a") as HTMLAnchorElement;
    expect(pty.textContent).toBe("PTY derselben Session öffnen");
    expect(pty.getAttribute("href")).toBe("/chat?resume=durable-1&profile=worker&pty=1");
  });
});
