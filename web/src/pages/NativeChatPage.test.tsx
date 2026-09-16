// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const gateway = vi.hoisted(() => {
  class MockGateway {
    static initialResumeResponse: Record<string, unknown> | null = null;
    static initialActivateBlock: Promise<void> | null = null;
    static initialActivateConsumed: (() => void) | null = null;
    static initialActivateError = false;
    static omitCreatedDurableIdentity = false;
    connectCalls = 0;
    stateHandler: ((state: string) => void) | null = null;
    handlers = new Map<string, (event: { type: string; session_id?: string; payload?: unknown }) => void>();
    requests: Array<{ method: string; params: Record<string, unknown> }> = [];
    snapshot: Record<string, unknown> | null = null;
    currentRuntimeId = "session-1";
    activateError = false;
    activateErrorOnce = false;
    resumeErrorOnce = false;
    activateBlock: Promise<void> | null = null;
    onActivateBlockConsumed: (() => void) | null = null;
    resumeResponse: Record<string, unknown> | null = null;
    lastDurableId = "stored-1";
    omitResumeMessages = false;
    constructor() {
      this.resumeResponse = MockGateway.initialResumeResponse;
      const initialDurable = this.resumeResponse?.session_key ?? this.resumeResponse?.stored_session_id;
      if (typeof initialDurable === "string" && initialDurable) this.lastDurableId = initialDurable;
      this.activateBlock = MockGateway.initialActivateBlock;
      this.onActivateBlockConsumed = MockGateway.initialActivateConsumed;
      this.activateError = MockGateway.initialActivateError;
    }
    submitBlock: Promise<void> | null = null;
    onSubmitBlockConsumed: (() => void) | null = null;
    approvalBlock: Promise<void> | null = null;
    approvalResponse: Record<string, unknown> = { resolved: 1 };
    onApprovalBlockConsumed: (() => void) | null = null;
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
    async connect() { this.connectCalls += 1; this.stateHandler?.("open"); }
    async request<T>(method: string, params: Record<string, unknown>) {
      this.requests.push({ method, params });
      if (method === "file.attach" && params.name === "fail.txt") throw new Error("upload failed");
      if (method === "prompt.submit" && this.submitBlock) {
        const block = this.submitBlock;
        this.submitBlock = null;
        this.onSubmitBlockConsumed?.();
        await block;
      }
      if (method === "approval.respond" && this.approvalBlock) {
        const block = this.approvalBlock;
        this.approvalBlock = null;
        this.onApprovalBlockConsumed?.();
        await block;
      }
      if (method === "session.activate" && this.activateBlock) {
        const block = this.activateBlock;
        this.activateBlock = null;
        this.onActivateBlockConsumed?.();
        await block;
      }
      if (method === "session.activate" && !this.resumeResponse && typeof params.session_id === "string" && params.session_id && params.session_id !== "session-1"
        && this.requests.filter(({ method: requestMethod }) => requestMethod === "session.activate").length === 1) {
        this.lastDurableId = params.session_id;
      }
      if (method === "session.activate" && (this.activateError || this.activateErrorOnce)) {
        this.activateErrorOnce = false;
        throw new Error("activate failed");
      }
      if (method === "session.resume" && this.resumeErrorOnce) {
        this.resumeErrorOnce = false;
        throw new Error("resume failed");
      }
      if (method === "prompt.submit" && params.text === "failed prompt" && this.requests.filter(({ method: requestMethod, params: requestParams }) => requestMethod === "prompt.submit" && requestParams.text === "failed prompt").length === 1) throw new Error("submit failed");
      if (method === "session.interrupt" && this.interrupt) await this.interrupt;
      if (method === "session.activate" || method === "session.resume") {
        const runtimeId = this.resumeResponse?.session_id
          ?? (params.session_id === "session-1" ? this.currentRuntimeId : "runtime-1");
        const response: Record<string, unknown> = { session_id: runtimeId, stored_session_id: this.lastDurableId, messages: [{ row_id: 7, id: 7, role: "user", text: "previous prompt" }, { row_id: 8, id: 8, role: "assistant", content: "previous answer" }], ...this.snapshot, ...this.resumeResponse };
        if (this.omitResumeMessages) delete response.messages;
        return response as T;
      }
      if (method === "session.branch") return { session_id: "branch-runtime", stored_session_id: "branch-durable", title: "Branch" } as T;
      if (method === "complete.slash") return { items: [{ display: "/help", text: "/help" }], replace_from: 0 } as T;
      if (method === "model.options") return { providers: [{ slug: "openai-codex", models: ["gpt-5.6-luna", "gpt-5.6-sol"] }, { slug: "openrouter", models: ["minimax/minimax-m3:free"] }] } as T;
      if (method === "approval.respond") return this.approvalResponse as T;
      const response = (method === "session.create" ? { session_id: "session-1", stored_session_id: "stored-1" } : { status: "streaming" }) as T;
      if (method === "session.create" && MockGateway.omitCreatedDurableIdentity) {
        const created = response as unknown as Record<string, unknown>;
        delete created.stored_session_id;
        delete created.session_key;
      }
      return response;
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
const profileScopeState = vi.hoisted(() => {
  let current = "thai-profile";
  const listeners = new Set<() => void>();
  return {
    get current() {
      return current;
    },
    set current(val: string) {
      current = val;
      listeners.forEach((listener) => listener());
    },
    reset() {
      current = "thai-profile";
      listeners.clear();
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
});

vi.mock("@/contexts/useProfileScope", async () => {
  const { useEffect, useState } = await import("react");
  return {
    useProfileScope: () => {
      const [profile, setProfile] = useState(profileScopeState.current);
      useEffect(() => {
        return profileScopeState.subscribe(() => {
          setProfile(profileScopeState.current);
        });
      }, []);
      return {
        profile,
        currentProfile: profile,
        profiles: ["thai-profile", "isolated-profile"],
        setProfile: (next: string) => {
          profileScopeState.current = next;
        },
      };
    },
  };
});
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

import NativeChatPage, { buildInflightFallbackKey, buildScopedIdentityKey, shouldMergeIdentityLessInflight, shouldReleaseQueueDrain, shouldRestoreStopTarget, shouldSubmitComposerKey, validateDurableIdentityResponse } from "./NativeChatPage";
import {
  nativeChatModelChoices,
  nativeChatSessionCreateParams,
} from "@/lib/native-chat-routing";
import { getArtifactStorageKey, makeArtifactId } from "@/lib/artifact-storage";

describe("NativeChatPage", () => {
  let root: Root;
  let host: HTMLDivElement;

  beforeEach(() => {
    profileScopeState.reset();
    gateway.MockGateway.initialResumeResponse = null;
    gateway.MockGateway.initialActivateBlock = null;
    gateway.MockGateway.initialActivateConsumed = null;
    gateway.MockGateway.initialActivateError = false;
    gateway.MockGateway.omitCreatedDurableIdentity = false;
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    localStorage.clear();
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    localStorage.clear();
  });

  it("blocks post-retirement unbound starts and controls without a new prompt", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "retired-control-turn" });
      gateway.instance?.emit("message.delta", { turn_id: "retired-control-turn", text: "retired control stream" });
      gateway.instance?.emit("session.info", { status: "idle", running: false, turn_id: "retired-control-turn" });
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("approval.request", { request_id: "stale-approval", command: "stale", choices: ["allow"] });
    });
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).toBeNull();
    expect(host.querySelector("button[data-choice='allow']")).toBeNull();
  });

  it("does not let identity-less idle session info retire an active turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "active-info-turn" });
    });
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).not.toBeNull();
    await act(async () => {
      gateway.instance?.emit("session.info", { status: "idle" });
    });
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).not.toBeNull();
  });

  it("does not let stale message identity retire a newer same-turn replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "replacement-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { turn_id: "replacement-turn", message_id: "old-message", text: "old answer" });
      gateway.instance?.emit("message.start", { turn_id: "replacement-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { turn_id: "replacement-turn", message_id: "new-message", text: "new answer" });
    });
    await vi.waitFor(() => {
      const rows = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
      const row = rows.find((candidate) => candidate.textContent?.includes("new answer"));
      expect(row).not.toBeUndefined();
    });
    await act(async () => gateway.instance?.emit("session.info", {
      status: "idle",
      running: false,
      turn_id: "replacement-turn",
      message_id: "previous-unretired-message",
    }));
    const currentReplacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((candidate) => candidate.textContent?.includes("new answer"));
    expect(currentReplacement?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("accepts a longer same-turn replacement answer", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "long-replacement-turn", message_id: "long-old" });
      gateway.instance?.emit("message.delta", { turn_id: "long-replacement-turn", message_id: "long-old", text: "same prefix" });
      gateway.instance?.emit("message.start", { turn_id: "long-replacement-turn", message_id: "long-new" });
      gateway.instance?.emit("message.delta", { turn_id: "long-replacement-turn", message_id: "long-new", text: "same prefix with new branch" });
    });
    expect(host.textContent).toContain("same prefix with new branch");
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).not.toBeNull();
  });

  it("attaches a durable error without clearing an unrelated active stream", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "session-1",
      stored_session_id: "stored-1",
      running: false,
      messages: [
        { row_id: 1, id: 1, role: "user", text: "old prompt" },
        { row_id: 2, id: 2, message_id: "durable-error-message", turn_id: "durable-error-turn", role: "assistant", text: "old answer", streaming: false },
      ],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=stored-1"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("old answer"));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "active-error-turn", message_id: "active-error-message" });
      gateway.instance?.emit("message.delta", { turn_id: "active-error-turn", message_id: "active-error-message", text: "active stream" });
    });
    await act(async () => gateway.instance?.emit("error", {
      turn_id: "durable-error-turn",
      message_id: "durable-error-message",
      error: "durable failure",
    }));

    const active = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((candidate) => candidate.textContent?.includes("active stream"));
    const durable = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((candidate) => candidate.textContent?.includes("old answer"));
    expect(active?.getAttribute("data-message-streaming")).toBe("true");
    expect(durable?.textContent).toContain("durable failure");
  });

  it("attaches a durable completion error without clearing an unrelated active stream", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "session-1",
      stored_session_id: "stored-1",
      running: false,
      messages: [
        { row_id: 1, id: 1, role: "user", text: "old completion prompt" },
        { row_id: 2, id: 2, message_id: "durable-completion-message", turn_id: "durable-completion-turn", role: "assistant", text: "old completion answer", streaming: false },
      ],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=stored-1"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("old completion answer"));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "active-completion-turn", message_id: "active-completion-message" });
      gateway.instance?.emit("message.delta", { turn_id: "active-completion-turn", message_id: "active-completion-message", text: "active completion stream" });
    });
    await act(async () => gateway.instance?.emit("message.complete", {
      turn_id: "durable-completion-turn",
      message_id: "durable-completion-message",
      error: "durable completion failure",
    }));

    const active = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((candidate) => candidate.textContent?.includes("active completion stream"));
    const durable = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((candidate) => candidate.textContent?.includes("old completion answer"));
    expect(active?.getAttribute("data-message-streaming")).toBe("true");
    expect(durable?.textContent).toContain("durable completion failure");
  });

  it("accepts a new-turn approval before message.start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "control-current-turn", message_id: "control-current-message" });
      gateway.instance?.emit("message.delta", { turn_id: "control-current-turn", message_id: "control-current-message", text: "current stream" });
    });
    await act(async () => gateway.instance?.emit("approval.request", {
      request_id: "new-control-approval",
      command: "new control command",
      choices: ["allow", "deny"],
      turn_id: "control-next-turn",
      message_id: "control-next-message",
    }));

    expect(host.textContent).toContain("new control command");
    expect(host.querySelector("button[data-choice='allow']")).toBeNull();
    expect(host.querySelector("button[data-choice='deny']")).not.toBeNull();
  });

  it("rebinds a same-turn replacement when the active row lacks messageId", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "unbound-active-turn" });
      gateway.instance?.emit("message.delta", { turn_id: "unbound-active-turn", text: "old unbound answer" });
      gateway.instance?.emit("message.start", { turn_id: "unbound-active-turn", message_id: "replacement-message" });
      gateway.instance?.emit("message.delta", { turn_id: "unbound-active-turn", message_id: "replacement-message", text: "new replacement answer" });
    });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(2);
    expect(host.textContent).toContain("old unbound answer");
    expect(host.textContent).toContain("new replacement answer");
    expect(assistants.at(-1)?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("rebinds a delta-only replacement when the active row lacks messageId", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "delta-replacement-turn" });
      gateway.instance?.emit("message.delta", { turn_id: "delta-replacement-turn", text: "old delta answer" });
      gateway.instance?.emit("message.delta", { turn_id: "delta-replacement-turn", message_id: "delta-new-message", text: "new delta answer" });
    });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(2);
    expect(host.textContent).toContain("old delta answer");
    expect(host.textContent).toContain("new delta answer");
    expect(assistants.at(-1)?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("does not let stale turn-only deltas authorize an empty replacement terminal", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "replacement-proof-turn", message_id: "replacement-proof-old" });
      gateway.instance?.emit("message.delta", { turn_id: "replacement-proof-turn", message_id: "replacement-proof-old", text: "old replacement answer" });
      gateway.instance?.emit("message.start", { turn_id: "replacement-proof-turn", message_id: "replacement-proof-new", text: "new replacement answer" });
    });
    await vi.waitFor(() => expect(host.textContent).toContain("new replacement answer"));

    await act(async () => gateway.instance?.emit("message.delta", {
      turn_id: "replacement-proof-turn",
      text: "old replacement answer",
    }));
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "replacement-proof-turn" }));

    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((candidate) => candidate.textContent?.includes("new replacement answer"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("merges identity-less inflight snapshots only for exact non-empty continuity", () => {
    expect(shouldMergeIdentityLessInflight("", "", "same prompt", "same prompt")).toBe(false);
    expect(shouldMergeIdentityLessInflight("partial", "partial plus", "same prompt", "same prompt")).toBe(false);
    expect(shouldMergeIdentityLessInflight("partial", "partial", "same prompt", "same prompt")).toBe(true);
  });

  it("keeps inflight fallback keys distinct when values contain the delimiter", () => {
    expect(buildInflightFallbackKey("scope", "user\u001f", "status", "error"))
      .not.toBe(buildInflightFallbackKey("scope", "user", "\u001fstatus", "error"));
  });

  it("does not promote an unrelated alias from a mixed durable response", () => {
    const validRekey = validateDurableIdentityResponse(
      "canonical-new",
      ["legacy-known", "unrelated-alias"],
      ["canonical-old", "legacy-known"],
    );
    expect(validRekey.accepted).toBe(true);
    expect(validRekey.canonicalId).toBe("canonical-new");
    expect(validRekey.validatedIds).toEqual(expect.arrayContaining(["canonical-new", "legacy-known"]));
    expect(validRekey.validatedIds).not.toContain("unrelated-alias");
  });

  it("fails closed when no durable response alias intersects the known set", () => {
    expect(validateDurableIdentityResponse("unknown-canonical", ["unknown-legacy"], ["known"])).toEqual({
      accepted: false,
      canonicalId: null,
      validatedIds: [],
    });
  });

  it("fails closed when a newly created response has no durable identity", async () => {
    gateway.MockGateway.omitCreatedDurableIdentity = true;
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='chat-error']")?.textContent ?? "").toContain("no durable session identity"));
  });

  it("does not restore a Stop target after a newer turn takes ownership", () => {
    expect(shouldRestoreStopTarget(
      { sessionGeneration: 4, requestGeneration: 7, assistantId: "assistant-old", turnId: "turn-old", turnGeneration: 11 },
      { sessionGeneration: 4, requestGeneration: 7, assistantId: "assistant-new", turnId: "turn-new", turnGeneration: 12 },
    )).toBe(false);
    expect(shouldRestoreStopTarget(
      { sessionGeneration: 4, requestGeneration: 7, assistantId: "assistant-old", turnId: "turn-old", turnGeneration: 11 },
      { sessionGeneration: 4, requestGeneration: 7, assistantId: "assistant-old", turnId: "turn-old", turnGeneration: 11 },
    )).toBe(true);
  });

  it("does not release a queue drain owned by an older session generation", () => {
    expect(shouldReleaseQueueDrain(4, 5)).toBe(false);
    expect(shouldReleaseQueueDrain(4, 4)).toBe(true);
  });

  it("keeps scoped identity keys collision-safe for delimiter-bearing values", () => {
    expect(buildScopedIdentityKey("a", "b:c")).not.toBe(buildScopedIdentityKey("a:b", "c"));
  });

  it("does not let duplicate event IDs raise the page sequence watermark", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "seq-message" , event_id: "same" }, "session-1", 5);
      gateway.instance?.emit("message.delta", { message_id: "seq-message", text: "duplicate", event_id: "same" }, "session-1", 9);
      gateway.instance?.emit("message.delta", { message_id: "seq-message", text: "legitimate", event_id: "next" }, "session-1", 6);
    });
    expect(host.textContent).toContain("legitimate");
    expect(host.textContent).not.toContain("duplicate");
  });

  it("admits an empty delta-first replacement while Stop is pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "empty-delta-old" });
      gateway.instance?.emit("message.delta", { turn_id: "empty-delta-old", text: "old empty-delta stream" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "empty-delta-old" }));
    await act(async () => gateway.instance?.emit("message.delta", { text: "" }));
    const replacement = host.querySelector<HTMLElement>("[data-message-role='assistant'][data-message-streaming='true']");
    expect(replacement).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.complete", { text: "" }));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("false");
    releaseInterrupt();
    await act(async () => { await Promise.resolve(); });
  });

  it("preserves an identified Stop fence after its matching terminal", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "identified-stop-turn", message_id: "identified-stop-message" });
      gateway.instance?.emit("message.delta", { turn_id: "identified-stop-turn", message_id: "identified-stop-message", text: "old identified stream" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "identified-stop-turn", message_id: "identified-stop-message", text: "old identified stream" }));
    releaseInterrupt();
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "new unbound stream" });
    });
    await act(async () => gateway.instance?.emit("message.complete", { text: "old identified stream" }));
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("new unbound stream"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("resyncs after a connection opens while Stop is pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "stop-resync-turn" });
      gateway.instance?.emit("message.delta", { turn_id: "stop-resync-turn", text: "stop resync stream" });
    });
    const initialActivateCount = () => gateway.instance?.requests.filter(({ method }) => method === "session.activate").length ?? 0;
    const beforeStopActivateCount = initialActivateCount();
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); });
    releaseInterrupt();
    await vi.waitFor(() => expect(initialActivateCount()).toBeGreaterThan(beforeStopActivateCount));
  });

  it("replays sequence-reset events that arrive while resync snapshot is pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "before-resync", event_id: "before-start" }, "session-1", 10);
      gateway.instance?.emit("message.delta", { message_id: "before-resync", text: "before resync", event_id: "before-delta" }, "session-1", 11);
    });
    let releaseActivate!: () => void;
    let activateConsumed!: () => void;
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = () => activateConsumed();
    const consumed = new Promise<void>((resolve) => { activateConsumed = resolve; });
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); });
    await consumed;
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "during-resync", event_id: "during-start" }, "session-1", 1);
      gateway.instance?.emit("message.delta", { message_id: "during-resync", text: "during resync", event_id: "during-delta" }, "session-1", 2);
    });
    await act(async () => {
      releaseActivate();
      await Promise.resolve();
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(host.textContent).toContain("during resync"));
  });

  it("keeps newer-turn events when a submit invalidates the pending resync barrier", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    let releaseActivate!: () => void;
    let activateConsumed!: () => void;
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = () => activateConsumed();
    const consumed = new Promise<void>((resolve) => { activateConsumed = resolve; });
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); });
    await consumed;

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "newer prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      await Promise.resolve();
    });
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "newer-turn", message_id: "newer-message" }, "session-1");
      gateway.instance?.emit("message.delta", { turn_id: "newer-turn", message_id: "newer-message", text: "newer event survived" }, "session-1");
    });
    await act(async () => {
      releaseActivate();
      await Promise.resolve();
      await Promise.resolve();
    });
    await vi.waitFor(() => expect(host.textContent).toContain("newer event survived"));
  });

  it("discards a resync snapshot that belongs to a closed connection epoch", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    let releaseActivate!: () => void;
    let activateConsumed!: () => void;
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = () => activateConsumed();
    const consumed = new Promise<void>((resolve) => { activateConsumed = resolve; });
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); });
    await consumed;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    gateway.instance!.snapshot = { session_id: "session-1", messages: [{ row_id: 901, role: "assistant", text: "stale closed-connection snapshot" }] };
    await act(async () => { releaseActivate(); await Promise.resolve(); });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.textContent).not.toContain("stale closed-connection snapshot");
  });

  it("replays a session-bound approval received while the initial snapshot is pending", async () => {
    let releaseActivate!: () => void;
    let activateConsumed!: () => void;
    gateway.MockGateway.initialActivateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    const consumed = new Promise<void>((resolve) => { activateConsumed = resolve; });
    gateway.MockGateway.initialActivateConsumed = () => activateConsumed();
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=resume-old"] }, createElement(NativeChatPage))));
    await consumed;
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-during-attach", command: "rm during-attach", choices: ["once", "deny"] }, "runtime-1"));
    await act(async () => { releaseActivate(); await Promise.resolve(); await Promise.resolve(); });
    await vi.waitFor(() => expect(host.textContent).toContain("rm during-attach"));
  });
  it("does not fall back to resume after initial activation is invalidated", async () => {
    let releaseActivate!: () => void;
    gateway.MockGateway.initialActivateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.MockGateway.initialActivateError = true;
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=resume-old"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.activate")).toBe(true));
    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    const oldResumeCount = gateway.instance?.requests.filter(({ method, params }) => method === "session.resume" && params.session_id === "resume-old").length ?? 0;
    await act(async () => { releaseActivate(); await Promise.resolve(); });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(gateway.instance?.requests.filter(({ method, params }) => method === "session.resume" && params.session_id === "resume-old").length).toBe(oldResumeCount);
  });
  it("rejects an initial resume response without durable aliases", async () => {
    gateway.MockGateway.initialResumeResponse = { session_id: "runtime-no-alias", stored_session_id: undefined, session_key: undefined, messages: [] };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=requested-key"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='chat-error']")?.textContent).toContain("no durable session identity"));
  });

  it("accepts a new canonical session_key when a known legacy alias matches", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-canonical-old",
      session_key: "canonical-old",
      stored_session_id: "legacy-alias",
      running: true,
      inflight: { user: "canonical prompt", assistant: "before canonical", streaming: true },
      messages: [],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=canonical-old"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("before canonical"));
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = {
      session_id: "runtime-canonical-new",
      session_key: "canonical-new",
      stored_session_id: "legacy-alias",
      running: true,
      inflight: { user: "canonical prompt", assistant: "before canonical continued", streaming: true },
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await vi.waitFor(() => expect(host.textContent).toContain("before canonical continued"));
    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
  });

  it("rejects a durable response without a previously validated alias", async () => {
    gateway.MockGateway.initialResumeResponse = { session_id: "runtime-alias-old", stored_session_id: "known-alias", session_key: "canonical-key", messages: [] };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=canonical-key"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-alias-new", stored_session_id: "unknown-alias", session_key: "conflicting-key", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='chat-error']")?.textContent ?? "").toContain("different durable session"));
  });

  it("does not retire a non-streaming interim snapshot assistant", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-interim-history",
      stored_session_id: "interim-history",
      messages: [{ message_id: "interim-message", role: "assistant", text: "interim answer", interim: true, streaming: false }],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=interim-history"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='transcript-message'][data-message-id='interim-message']")).not.toBeNull());
    await act(async () => gateway.instance?.emit("message.complete", { message_id: "interim-message", text: "finalized interim answer" }, "runtime-interim-history"));
    expect(host.textContent).toContain("finalized interim answer");
  });

  it("keeps a non-streaming snapshot assistant alive when running inflight has the same identity", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-inflight-snapshot",
      stored_session_id: "inflight-snapshot",
      running: true,
      messages: [
        { row_id: 71, role: "user", text: "inflight prompt" },
        { row_id: 72, message_id: "inflight-message", turn_id: "inflight-turn", role: "assistant", text: "partial", streaming: false },
      ],
      inflight: {
        user: "inflight prompt",
        assistant: "partial",
        message_id: "inflight-message",
        turn_id: "inflight-turn",
        streaming: true,
      },
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=inflight-snapshot"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("partial"));
    await act(async () => gateway.instance?.emit("message.delta", {
      message_id: "inflight-message",
      turn_id: "inflight-turn",
      text: " continued",
    }, "runtime-inflight-snapshot"));
    expect(host.textContent).toContain("continued");
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
    expect(host.querySelector("[data-message-role='assistant']")?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("allows an exact durable turn error despite the historical turn fence", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-durable-turn-error",
      stored_session_id: "durable-turn-error",
      messages: [{ turn_id: "historical-turn", message_id: "historical-message", role: "assistant", text: "historical partial" }],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-turn-error"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='transcript-message'][data-message-id='historical-message']")).not.toBeNull());
    await act(async () => gateway.instance?.emit("error", { turn_id: "historical-turn", message_id: "historical-message", error: "historical provider failure" }, "runtime-durable-turn-error"));
    const row = host.querySelector<HTMLElement>("[data-slot='transcript-message'][data-message-id='historical-message']");
    expect(row?.getAttribute("data-message-error")).toBe("true");
    expect(row?.textContent).toContain("historical provider failure");
  });

  it("uses a durable message_id when a snapshot row has no row_id", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-message-id",
      stored_session_id: "message-id-durable",
      messages: [{ message_id: "durable-message-only", role: "assistant", text: "durable message-id answer" }],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=message-id-durable"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='transcript-message'][data-message-id='durable-message-only']")).not.toBeNull());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("error", { message_id: "durable-message-only", error: "durable message-id failure" }, "runtime-message-id"));
    await vi.waitFor(() => expect(host.querySelector<HTMLElement>("[data-slot='transcript-message'][data-message-id='durable-message-only']")?.getAttribute("data-message-error")).toBe("true"));
    const row = host.querySelector<HTMLElement>("[data-slot='transcript-message'][data-message-id='durable-message-only']");
    expect(row?.textContent).toContain("durable message-id failure");
  });

  it("resolves a hydrated streaming row on an empty durable completion", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-empty-completion",
      stored_session_id: "empty-completion",
      messages: [{ message_id: "empty-completion-message", role: "assistant", text: "hydrated partial", streaming: true }],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=empty-completion"] }, createElement(NativeChatPage))));
    const selector = "[data-slot='transcript-message'][data-message-id='empty-completion-message']";
    await vi.waitFor(() => expect(host.querySelector<HTMLElement>(selector)?.getAttribute("data-message-streaming")).toBe("true"));
    await act(async () => gateway.instance?.emit("message.complete", { message_id: "empty-completion-message", text: "" }, "runtime-empty-completion"));
    const row = host.querySelector<HTMLElement>(selector);
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
    expect(row?.getAttribute("data-message-streaming")).toBe("false");
    expect(row?.textContent).toContain("hydrated partial");
  });

  it("does not create a duplicate empty row for an unmatched durable completion", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-unmatched-empty",
      stored_session_id: "unmatched-empty",
      messages: [{ row_id: 81, role: "user", text: "hydrated prompt" }],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=unmatched-empty"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("hydrated prompt"));
    await act(async () => gateway.instance?.emit("message.complete", { message_id: "missing-completion-message", text: "" }, "runtime-unmatched-empty"));
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(0);
  });

  it("reconciles matching delta and completion identity with a hydrated snapshot row", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-same-turn",
      stored_session_id: "same-turn-durable",
      messages: [
        { row_id: 41, role: "user", text: "same-turn prompt" },
        { row_id: 42, message_id: "same-turn-message", turn_id: "same-turn", role: "assistant", text: "partial", streaming: true },
      ],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=same-turn-durable"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "same-turn", message_id: "same-turn-message" }, "runtime-same-turn");
      gateway.instance?.emit("message.delta", { turn_id: "same-turn", message_id: "same-turn-message", text: "partial" }, "runtime-same-turn");
    });
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
    expect(host.querySelector("[data-message-role='assistant']")?.textContent).toContain("partial");
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "same-turn", message_id: "same-turn-message" }, "runtime-same-turn"));
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
    expect(host.querySelector("[data-message-role='assistant']")?.getAttribute("data-message-streaming")).toBe("false");
  });

  it("preserves durable message error and interim metadata on resume", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-metadata",
      stored_session_id: "metadata-durable",
      messages: [
        { row_id: 11, role: "user", text: "metadata prompt" },
        { row_id: 12, role: "assistant", text: "partial answer", error: "provider failed", interim: true, streaming: true },
      ],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=metadata-durable"] }, createElement(NativeChatPage))));
    const assistant = await vi.waitFor(() => {
      const row = host.querySelector<HTMLElement>("[data-message-role='assistant']");
      expect(row).not.toBeNull();
      return row!;
    });
    expect(assistant.dataset.messageError).toBe("true");
    expect(assistant.dataset.messageInterim).toBe("true");
    expect(assistant.dataset.messageStreaming).toBe("true");
    expect(assistant.textContent).toContain("provider failed");
  });

  it("rejects an initial resume response for a different durable session", async () => {
    gateway.MockGateway.initialResumeResponse = { session_id: "runtime-other", stored_session_id: "other-stored", session_key: "other-key", messages: [] };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=requested-key"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='chat-error']")?.textContent).toContain("different durable session"));
    expect(host.textContent).not.toContain("previous answer");
  });


  it("prefers session_key as durable identity when both session keys are returned", async () => {
    gateway.MockGateway.initialResumeResponse = { session_id: "runtime-key", stored_session_id: "legacy-stored", session_key: "durable-key", messages: [] };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=legacy-stored"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const html = `<!doctype html><html><body>${"x".repeat(180)}</body></html>`;
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "key-message" }, "runtime-key");
      gateway.instance?.emit("message.delta", { text: `\`\`\`html\n${html}\n\`\`\``, message_id: "key-message" }, "runtime-key");
      gateway.instance?.emit("message.complete", { message_id: "key-message" }, "runtime-key");
    });
    const pin = await vi.waitFor(() => {
      const button = host.querySelector<HTMLButtonElement>('button[aria-label="Pin artifact"]');
      expect(button).not.toBeNull();
      return button!;
    });
    await act(async () => pin.click());
    const stored = JSON.parse(localStorage.getItem(getArtifactStorageKey("thai-profile")) ?? "[]") as Array<{ sessionId: string }>;
    expect(stored[0]?.sessionId).toBe("durable-key");
    expect(stored[0]?.sessionId).not.toBe("runtime-key");
  });

  it("retires identified historical messages hydrated from a durable snapshot", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-history",
      stored_session_id: "history-durable",
      messages: [
        { id: "historic-message", role: "assistant", text: "historic answer", turn_id: "historic-turn" },
      ],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=history-durable"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("historic answer"));
    await act(async () => gateway.instance?.emit("message.delta", { message_id: "historic-message", turn_id: "historic-turn", text: "late historic mutation" }, "runtime-history"));
    await act(async () => gateway.instance?.emit("message.complete", { message_id: "historic-message", text: "late historic completion" }, "runtime-history"));
    expect(host.textContent).not.toContain("late historic mutation");
    expect(host.textContent).not.toContain("late historic completion");
  });

  it("keeps canonical session scope when reconnect returns only a legacy alias", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-canonical",
      session_key: "canonical-k",
      stored_session_id: "legacy-l",
      messages: [],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=canonical-k"] }, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "retired-message" }, "runtime-canonical");
      gateway.instance?.emit("message.delta", { message_id: "retired-message", text: "retired answer" }, "runtime-canonical");
      gateway.instance?.emit("message.complete", { message_id: "retired-message" }, "runtime-canonical");
    });
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-reconnected", stored_session_id: "legacy-l", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("message.delta", { message_id: "retired-message", text: "resurrected legacy replay" }, "runtime-reconnected"));
    expect(host.textContent).not.toContain("resurrected legacy replay");
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

  it("uses durable stored session identity for artifact pins", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const html = `<!doctype html><html><head><title>Durable artifact</title></head><body>${"x".repeat(180)}</body></html>`;
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: `\`\`\`html\n${html}\n\`\`\`` });
      gateway.instance?.emit("message.complete");
    });

    const pin = await vi.waitFor(() => {
      const button = host.querySelector<HTMLButtonElement>('button[aria-label="Pin artifact"]');
      expect(button).not.toBeNull();
      return button!;
    });
    await act(async () => pin.click());
    const stored = JSON.parse(localStorage.getItem(getArtifactStorageKey("thai-profile")) ?? "[]") as Array<{ sessionId: string }>;
    expect(stored).toHaveLength(1);
    expect(stored[0]?.sessionId).toBe("stored-1");
    expect(stored[0]?.sessionId).not.toBe("session-1");
  });

  it("rekeys pinned artifacts when reconnect promotes a canonical session key", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const html = `<!doctype html><html><head><title>Rekey artifact</title></head><body>${"x".repeat(180)}</body></html>`;
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: `\`\`\`html\n${html}\n\`\`\`` });
      gateway.instance?.emit("message.complete");
    });
    const pin = await vi.waitFor(() => {
      const button = host.querySelector<HTMLButtonElement>('button[aria-label="Pin artifact"]');
      expect(button).not.toBeNull();
      return button!;
    });
    await act(async () => pin.click());
    const before = JSON.parse(localStorage.getItem(getArtifactStorageKey("thai-profile")) ?? "[]") as Array<Record<string, unknown>>;
    expect(before).toHaveLength(1);
    expect(before[0]?.sessionId).toBe("stored-1");

    gateway.instance!.resumeResponse = {
      session_id: "session-1",
      session_key: "canonical-session",
      stored_session_id: "stored-1",
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method }) => method === "session.activate")).not.toHaveLength(0));
    await vi.waitFor(() => {
      const after = JSON.parse(localStorage.getItem(getArtifactStorageKey("thai-profile")) ?? "[]") as Array<Record<string, unknown>>;
      expect(after[0]?.sessionId).toBe("canonical-session");
      expect(after[0]?.id).toBe(makeArtifactId(
        "canonical-session",
        String(before[0]?.kind) as "html",
        String(before[0]?.language),
        String(before[0]?.title),
        String(before[0]?.code),
      ));
    });
  });

  it("keeps resync error state when activate and resume both fail", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    gateway.instance!.activateError = true;
    gateway.instance!.resumeErrorOnce = true;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method }) => method === "session.activate").length).toBeGreaterThan(0));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.querySelector("[data-slot='chat-notices']")?.getAttribute("data-resync-state")).toBe("error");
    expect(host.textContent).toContain("resume failed");
  });

  it("clears a stale resync error after a later active snapshot succeeds", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    gateway.instance!.activateError = true;
    gateway.instance!.resumeErrorOnce = true;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(host.querySelector("[data-slot='chat-notices']")?.getAttribute("data-resync-state")).toBe("error"));
    gateway.instance!.activateError = false;
    gateway.instance!.resumeResponse = {
      session_id: "session-1",
      stored_session_id: "stored-1",
      running: true,
      inflight: { user: "active prompt", assistant: "active answer", streaming: true },
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(host.textContent).toContain("active answer"));
    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
  });

  it("does not fall back to resume after resync is invalidated", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    let releaseActivate!: () => void;
    let activateConsumed!: () => void;
    const consumed = new Promise<void>((resolve) => { activateConsumed = resolve; });
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = activateConsumed;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await consumed;
    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.activateErrorOnce = true;
    releaseActivate();
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(gateway.instance?.requests.some(({ method, params }) => method === "session.resume" && params.session_id === "session-1")).toBe(false);
  });

  it("keeps chat text controls at a mobile-safe size while staying compact on desktop", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea");
    expect(textarea?.className).toContain("text-[16px]");
    expect(textarea?.className).toContain("sm:text-sm");

    for (const select of Array.from(host.querySelectorAll("select"))) {
      expect(select.className).toContain("text-[16px]");
      expect(select.className).toContain("sm:text-xs");
    }
  });

  it("exposes a voice control and reports a clear fallback when recording is unavailable", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const record = host.querySelector<HTMLButtonElement>("button[aria-label='Record voice']");
    expect(record).toBeTruthy();
    await act(async () => record?.click());
    expect(host.querySelector("[data-slot='voice-error']")?.textContent).toContain("Voice input is not available");
  });

  it("hides sender names from message bubbles while preserving accessible labels", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "user message");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "assistant message" });
      gateway.instance?.emit("message.complete");
    });

    const userMessage = host.querySelector<HTMLElement>("[data-message-role='user']");
    const assistantMessage = host.querySelector<HTMLElement>("[data-message-role='assistant']");
    expect(userMessage?.textContent).not.toContain("You");
    expect(assistantMessage?.textContent).not.toContain("Hermes");
    expect(userMessage?.querySelector(".mb-1")).toBeNull();
    expect(assistantMessage?.querySelector(".mb-1")).toBeNull();
    expect(userMessage?.getAttribute("aria-label")).toBe("Your message");
    expect(assistantMessage?.getAttribute("aria-label")).toBe("Hermes message");
  });

  it("rejects an empty late EOF from a stopped turn after an unbound replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "empty-late-old" });
      gateway.instance?.emit("message.delta", { turn_id: "empty-late-old", text: "old stream" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "new replacement" });
    });
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("new replacement"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
    await act(async () => gateway.instance?.emit("message.complete", { text: "" }));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("finishes a submitted identity-less turn on an empty EOF", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "submitted empty turn");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
    });
    await act(async () => gateway.instance?.emit("message.start"));
    const streaming = host.querySelector<HTMLElement>("[data-message-role='assistant'][data-message-streaming='true']");
    expect(streaming).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.complete", { text: "" }));
    expect(streaming?.getAttribute("data-message-streaming")).toBe("false");
  });

  it("keeps multiple prompt and response turns in chronological order", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    const submitTurn = async (prompt: string, answer: string) => {
      await act(async () => {
        setter?.call(textarea, prompt);
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      });
      await act(async () => {
        gateway.instance?.emit("message.start");
        gateway.instance?.emit("message.delta", { text: answer });
        gateway.instance?.emit("message.complete");
      });
    };

    await submitTurn("first prompt", "first answer");
    await submitTurn("second prompt", "second answer");

    const messages = Array.from(host.querySelectorAll<HTMLElement>("[data-slot='transcript-message']"));
    expect(messages.map((message) => message.dataset.messageRole)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(messages[0]?.textContent).toContain("first prompt");
    expect(messages[1]?.textContent).toContain("first answer");
    expect(messages[2]?.textContent).toContain("second prompt");
    expect(messages[3]?.textContent).toContain("second answer");
  });

  it("accepts an untagged backend-originated turn after a completed turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "first answer" });
      gateway.instance?.emit("message.complete");
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "auto answer" });
      gateway.instance?.emit("message.complete");
    });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(2);
    expect(assistants[0]?.textContent).toContain("first answer");
    expect(assistants[1]?.textContent).toContain("auto answer");
  });

  it("does not duplicate an unbound stream when a tagged start replay arrives", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "first part" });
      gateway.instance?.emit("message.start", { turn_id: "turn-1" });
    });

    expect(host.querySelectorAll("[data-slot='transcript-message']")).toHaveLength(1);
    await act(async () => {
      gateway.instance?.emit("message.delta", { text: " second part", turn_id: "turn-1" });
      gateway.instance?.emit("message.complete", { turn_id: "turn-1" });
    });
    const messages = Array.from(host.querySelectorAll<HTMLElement>("[data-slot='transcript-message']"));
    expect(messages).toHaveLength(1);
    expect(messages[0]?.textContent).toContain("first part second part");
  });

  it("does not duplicate a message-id-only turn replay after completion", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "message-only" });
      gateway.instance?.emit("message.delta", { text: "message-only answer", message_id: "message-only" });
      gateway.instance?.emit("message.complete", { message_id: "message-only" });
      gateway.instance?.emit("message.start", { message_id: "message-only" });
      gateway.instance?.emit("message.delta", { text: "message-only answer", message_id: "message-only" });
    });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(1);
    expect(assistants[0]?.textContent).toContain("message-only answer");
  });

  it("accepts a tagged replacement after an unbound stream when message identity changes", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", message_id: "old-message" });
      gateway.instance?.emit("message.start", { turn_id: "new-turn", message_id: "new-message" });
    });

    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(2);
    expect(host.textContent).toContain("old partial");
    await act(async () => {
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { turn_id: "new-turn", message_id: "new-message" });
    });
    expect(host.textContent).toContain("new answer");
  });

  it("blocks an unbound terminal before a tagged stream establishes its first delta", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "tagged-boundary-turn" }));
    const current = host.querySelector<HTMLElement>("[data-message-role='assistant'][data-message-streaming='true']");
    expect(current).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.complete", { text: "unbound boundary terminal" }));
    expect(current?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("does not let an unbound terminal clear a tagged stream before its first delta", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "tagged-current-turn" }));
    const current = host.querySelector<HTMLElement>("[data-message-role='assistant'][data-message-streaming='true']");
    expect(current).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.complete", { text: "unbound stale terminal" }));
    expect(current?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("splits a same-turn stream when its message identity changes", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "same-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old answer", turn_id: "same-turn", message_id: "old-message" });
      gateway.instance?.emit("message.start", { turn_id: "same-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "same-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { turn_id: "same-turn", message_id: "new-message" });
    });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(2);
    expect(host.textContent).toContain("old answer");
    expect(host.textContent).toContain("new answer");
  });

  it("routes an explicit replacement terminal away from an active unbound stream", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "old unbound before replacement" });
      gateway.instance?.emit("message.complete", { turn_id: "replacement-turn", message_id: "replacement-message", text: "new explicit terminal" });
    });
    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(2);
    expect(host.textContent).toContain("old unbound before replacement");
    expect(host.textContent).toContain("new explicit terminal");
  });

  it("rejects an old turn-only delta after a same-turn message replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "delta-fence-turn", message_id: "delta-fence-old" });
      gateway.instance?.emit("message.delta", { turn_id: "delta-fence-turn", message_id: "delta-fence-old", text: "old delta answer" });
      gateway.instance?.emit("message.start", { turn_id: "delta-fence-turn", message_id: "delta-fence-new" });
      gateway.instance?.emit("message.delta", { turn_id: "delta-fence-turn", message_id: "delta-fence-new", text: "new delta answer" });
    });
    await act(async () => gateway.instance?.emit("message.delta", { turn_id: "delta-fence-turn", text: "old delta answer" }));
    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants.at(-1)?.textContent).not.toContain("old delta answer");
    expect(assistants.at(-1)?.textContent).toContain("new delta answer");
  });

  it("admits content-proven turn-only events after a same-turn message replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "fenced-turn", message_id: "fenced-old" });
      gateway.instance?.emit("message.delta", { turn_id: "fenced-turn", message_id: "fenced-old", text: "old fenced answer" });
      gateway.instance?.emit("message.start", { turn_id: "fenced-turn", message_id: "fenced-new" });
      gateway.instance?.emit("message.delta", { turn_id: "fenced-turn", text: "new fenced answer" });
      gateway.instance?.emit("message.complete", { turn_id: "fenced-turn", text: "new fenced answer" });
    });
    expect(host.textContent).toContain("old fenced answer");
    expect(host.textContent).toContain("new fenced answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("accepts an empty turn-only EOF after a replacement delta with message_id", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "message-eof-turn", message_id: "message-eof-old" });
      gateway.instance?.emit("message.delta", { turn_id: "message-eof-turn", message_id: "message-eof-old", text: "old message eof" });
      gateway.instance?.emit("message.start", { turn_id: "message-eof-turn", message_id: "message-eof-new" });
      gateway.instance?.emit("message.delta", { turn_id: "message-eof-turn", message_id: "message-eof-new", text: "new message eof" });
      gateway.instance?.emit("message.complete", { turn_id: "message-eof-turn", text: "" });
    });
    expect(host.textContent).toContain("new message eof");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("accepts an empty EOF after a same-turn replacement has produced content", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "empty-eof-turn", message_id: "empty-eof-old" });
      gateway.instance?.emit("message.delta", { turn_id: "empty-eof-turn", message_id: "empty-eof-old", text: "old eof answer" });
      gateway.instance?.emit("message.start", { turn_id: "empty-eof-turn", message_id: "empty-eof-new" });
      gateway.instance?.emit("message.delta", { turn_id: "empty-eof-turn", text: "new eof answer" });
      gateway.instance?.emit("message.complete", { turn_id: "empty-eof-turn", text: "" });
    });
    expect(host.textContent).toContain("old eof answer");
    expect(host.textContent).toContain("new eof answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("accepts a replacement terminal that arrives before message.start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "terminal-old-turn", message_id: "terminal-old-message" });
      gateway.instance?.emit("message.delta", { text: "old terminal stream", turn_id: "terminal-old-turn", message_id: "terminal-old-message" });
      gateway.instance?.emit("message.complete", { text: "new terminal answer", turn_id: "terminal-new-turn", message_id: "terminal-new-message" });
    });

    expect(host.textContent).toContain("old terminal stream");
    expect(host.textContent).toContain("new terminal answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("rejects a late turn-only terminal after a same-turn message replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "shared-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { turn_id: "shared-turn", message_id: "old-message", text: "old shared stream" });
      gateway.instance?.emit("message.start", { turn_id: "shared-turn", message_id: "new-message" });
    });
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']")).at(-1);
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "shared-turn" }));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("accepts a replacement error that arrives before message.start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "error-old-turn", message_id: "error-old-message" });
      gateway.instance?.emit("message.delta", { text: "old error stream", turn_id: "error-old-turn", message_id: "error-old-message" });
      gateway.instance?.emit("error", { turn_id: "error-new-turn", message_id: "error-new-message", error: "replacement failed" });
    });

    expect(host.textContent).toContain("old error stream");
    expect(host.querySelector("[data-slot='chat-error']")?.textContent).toContain("replacement failed");
  });

  it("does not replay a rejected seq-less terminal after the replacement delta", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "first prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "first answer" });
      gateway.instance?.emit("message.complete", { event_id: "first-complete" });
    });
    await act(async () => {
      setter?.call(textarea, "replacement prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.complete", { text: "stale answer", event_id: "stale-complete" });
      gateway.instance?.emit("message.start", { event_id: "replacement-start" });
      gateway.instance?.emit("message.delta", { text: "replacement answer", event_id: "replacement-delta" });
      gateway.instance?.emit("message.complete", { text: "stale answer", event_id: "stale-complete" });
      gateway.instance?.emit("message.complete", { text: "replacement answer", event_id: "replacement-complete" });
    });

    expect(host.textContent).not.toContain("stale answer");
    expect(host.textContent).toContain("replacement answer");
  });

  it("cleans an error snapshot that omits running and inflight", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "status-error-turn", message_id: "status-error-message" });
      gateway.instance?.emit("message.delta", { turn_id: "status-error-turn", message_id: "status-error-message", text: "partial" });
    });
    gateway.instance!.snapshot = { session_id: "session-1", status: "error", error: "snapshot failure", messages: [] };
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); await Promise.resolve(); });
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.textContent).toContain("snapshot failure");
  });

  it("cleans an explicit ready session.info with matching turn identity", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "info-ready-turn" });
      gateway.instance?.emit("message.delta", { text: "info ready partial", turn_id: "info-ready-turn" });
      gateway.instance?.emit("session.info", { status: "ready", turn_id: "info-ready-turn" });
    });
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.querySelector("button[aria-label='Stop']")).toBeNull();
  });

  it("cleans an idle session.info with matching turn identity", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "info-idle-turn" });
      gateway.instance?.emit("message.delta", { text: "info idle partial", turn_id: "info-idle-turn" });
      gateway.instance?.emit("session.info", { status: "idle", turn_id: "info-idle-turn" });
    });

    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.querySelector("button[aria-label='Stop']")).toBeNull();
  });

  it("cleans an idle snapshot with status but without running", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "status-idle-turn", message_id: "status-idle-message" });
      gateway.instance?.emit("message.delta", { text: "status idle partial", turn_id: "status-idle-turn", message_id: "status-idle-message" });
    });
    gateway.instance!.snapshot = { session_id: "session-1", status: "idle", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.querySelector("button[aria-label='Stop']")).toBeNull();
    expect(host.textContent).not.toContain("status idle partial");
  });

  it("does not let an unbound old turn error terminate a new stream", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "new unbound stream" });
      gateway.instance?.emit("error", { kind: "turn", error_surface: "turn", error: "old turn failed" });
    });

    expect(host.textContent).toContain("new unbound stream");
    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
    expect(host.querySelector("button[aria-label='Stop']")).toBeTruthy();
  });

  it("surfaces a current unbound turn error after a submitted prompt", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "current unbound prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
      await Promise.resolve();
    });
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "current partial" });
      gateway.instance?.emit("error", { kind: "turn", error: "current turn failed" });
    });
    expect(host.textContent).toContain("current turn failed");
    expect(host.querySelector("[data-message-role='assistant'][data-message-error='true']")).not.toBeNull();
  });

  it("admits a no-identity delta-first replacement while Stop is pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "pending-old-turn" });
      gateway.instance?.emit("message.delta", { turn_id: "pending-old-turn", text: "old pending stream" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => gateway.instance?.emit("message.delta", { text: "replacement while Stop pending" }));
    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants.filter((row) => row.textContent?.includes("old pending stream"))).toHaveLength(1);
    expect(assistants.filter((row) => row.textContent?.includes("replacement while Stop pending"))).toHaveLength(1);
    releaseInterrupt();
    await act(async () => { await Promise.resolve(); });
  });

  it("keeps a retired Stop fence effective across canonical rekey", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "canonical-stop-old", message_id: "canonical-stop-message" });
      gateway.instance?.emit("message.delta", { turn_id: "canonical-stop-old", message_id: "canonical-stop-message", text: "canonical old stop stream" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { releaseInterrupt(); await Promise.resolve(); });
    await vi.waitFor(() => expect(host.querySelector("button[aria-label='Stop']")).toBeNull());

    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = {
      session_id: "runtime-canonical-rekey",
      session_key: "canonical-rekeyed",
      stored_session_id: "stored-1",
      messages: [],
      running: true,
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => {
      gateway.instance?.emit("message.start", {}, "runtime-canonical-rekey");
      gateway.instance?.emit("message.delta", { text: "canonical new unbound stream" }, "runtime-canonical-rekey");
    });
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("canonical new unbound stream"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
    await act(async () => gateway.instance?.emit("message.complete", { text: "canonical old stop stream" }, "runtime-canonical-rekey"));
    const currentReplacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("canonical new unbound stream"));
    expect(currentReplacement?.getAttribute("data-message-streaming")).toBe("true");
  });

  it("keeps an identified Stop fence through an unbound replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "identified-stop-turn", message_id: "identified-stop-message" });
      gateway.instance?.emit("message.delta", { turn_id: "identified-stop-turn", message_id: "identified-stop-message", text: "identified stop stream" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { releaseInterrupt(); await Promise.resolve(); });
    await vi.waitFor(() => expect(host.querySelector("button[aria-label='Stop']")).toBeNull());

    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "new unbound replacement" });
    });
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("new unbound replacement"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("true");
    await act(async () => gateway.instance?.emit("message.complete", { text: "identified stop stream" }));
    const currentReplacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("new unbound replacement"));
    expect(currentReplacement?.getAttribute("data-message-streaming")).toBe("true");
    expect(currentReplacement?.textContent).toContain("new unbound replacement");
  });

  it("allows an empty EOF for a start-first replacement while Stop is pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "empty-old-turn" });
      gateway.instance?.emit("message.delta", { turn_id: "empty-old-turn", text: "old empty-boundary stream" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "empty-old-turn" }));
    await act(async () => gateway.instance?.emit("message.start"));
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.getAttribute("data-message-streaming") === "true");
    expect(replacement).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.complete"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("false");
    releaseInterrupt();
    await act(async () => { await Promise.resolve(); });
  });

  it("allows a submitted delta-first replacement after Stop", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-stop-turn", message_id: "old-stop-message" });
      gateway.instance?.emit("message.delta", { turn_id: "old-stop-turn", message_id: "old-stop-message", text: "old" });
    });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']");
    expect(stop).not.toBeNull();
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => { stop!.click(); await Promise.resolve(); });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, "replacement prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => {
      gateway.instance?.emit("message.delta", { text: "replacement answer" });
      gateway.instance?.emit("message.complete", { text: "replacement answer" });
      releaseInterrupt();
      await Promise.resolve();
    });
    expect(host.textContent).toContain("replacement answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("blocks a late old terminal after a submitted delta-first replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "old answer" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "replacement after old");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => gateway.instance?.emit("message.delta", { text: "replacement partial" }));
    expect(host.textContent).toContain("replacement partial");
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.complete", { text: "old answer" }));
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).not.toBeNull();
  });

  it("preserves start-first replacement fencing and consumes one stale empty terminal", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "old start answer" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "start-first replacement");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => gateway.instance?.emit("message.start"));
    await act(async () => gateway.instance?.emit("message.delta", { text: "replacement stream" }));
    const replacement = () => Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("replacement stream"));
    expect(replacement()?.getAttribute("data-message-streaming")).toBe("true");
    await act(async () => gateway.instance?.emit("message.complete", { text: "old start answer" }));
    expect(replacement()?.getAttribute("data-message-streaming")).toBe("true");
    await act(async () => gateway.instance?.emit("message.complete", { text: "old start answer" }));
    expect(replacement()?.getAttribute("data-message-streaming")).toBe("true");
    await act(async () => gateway.instance?.emit("message.complete"));
    expect(replacement()?.getAttribute("data-message-streaming")).toBe("false");
  });

  it("does not swallow a text-bearing completion after a textless Stop", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start"));
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "replacement after textless stop");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => gateway.instance?.emit("message.start"));
    await act(async () => gateway.instance?.emit("message.delta", { text: "replacement after textless stop answer" }));
    await act(async () => gateway.instance?.emit("message.complete", { text: "replacement after textless stop answer" }));
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("replacement after textless stop answer"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("false");
  });

  it("accepts a replacement empty EOF when no stale terminal preceded it", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "old non-empty answer" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "replacement empty EOF");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => gateway.instance?.emit("message.start"));
    await act(async () => gateway.instance?.emit("message.delta", { text: "replacement empty EOF answer" }));
    await act(async () => gateway.instance?.emit("message.complete"));
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("replacement empty EOF answer"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("false");
  });

  it("does not fence an unbound replacement behind a tagged Stop target", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "tagged-stop-only" }));
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "replacement after tagged stop");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      await Promise.resolve();
    });
    await act(async () => gateway.instance?.emit("message.start"));
    await act(async () => gateway.instance?.emit("message.delta", { text: "replacement after tagged stop answer" }));
    await act(async () => gateway.instance?.emit("message.complete", { text: "replacement after tagged stop answer" }));
    const replacement = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .find((row) => row.textContent?.includes("replacement after tagged stop answer"));
    expect(replacement?.getAttribute("data-message-streaming")).toBe("false");
  });

  it("lets a new untagged stream complete after old Stop terminal arrives first", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "stop-old-turn", message_id: "stop-old-message" });
      gateway.instance?.emit("message.delta", { text: "old stream", turn_id: "stop-old-turn", message_id: "stop-old-message" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "stop-old-turn", message_id: "stop-old-message" }));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "new untagged stream" });
      gateway.instance?.emit("message.complete", { text: "new untagged answer" });
    });
    releaseInterrupt();
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("new untagged answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("does not let a late unbound terminal clear a replacement after its delta", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "old partial" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => gateway.instance?.emit("message.complete", { text: "old partial" }));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "replacement partial" });
      gateway.instance?.emit("message.complete", { text: "old partial" });
    });
    expect(host.textContent).toContain("replacement partial");
    expect(host.querySelector("[data-slot='turn-activity']")).not.toBeNull();
    releaseInterrupt();
    await act(async () => { await Promise.resolve(); });
  });

  it("keeps the retirement fence after Stop completes before an unbound replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "old completed partial" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => { await Promise.resolve(); });
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "replacement after stop" });
      gateway.instance?.emit("message.complete", { text: "old completed partial" });
    });
    expect(host.textContent).toContain("replacement after stop");
    expect(host.querySelector("[data-slot='turn-activity']")).not.toBeNull();
  });

  it("accepts a turn-only replacement terminal before its start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.complete", { text: "new turn answer", turn_id: "new-turn" });
    });

    expect(host.textContent).toContain("old partial");
    expect(host.textContent).toContain("new turn answer");
  });

  it("accepts a turn-only replacement error before its start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-error-turn", message_id: "old-error-message" });
      gateway.instance?.emit("message.delta", { text: "old error partial", turn_id: "old-error-turn", message_id: "old-error-message" });
      gateway.instance?.emit("error", { turn_id: "new-error-turn", error: "new turn failed" });
    });

    expect(host.textContent).toContain("old error partial");
    expect(host.querySelector("[data-slot='chat-error']")?.textContent).toContain("new turn failed");
  });

  it("splits a tagged stream when a delta changes its message identity", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "delta-turn", message_id: "delta-old" });
      gateway.instance?.emit("message.delta", { text: "delta old", turn_id: "delta-turn", message_id: "delta-old" });
      gateway.instance?.emit("message.delta", { text: "delta new", turn_id: "delta-turn", message_id: "delta-new" });
    });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(2);
    expect(host.textContent).toContain("delta old");
    expect(host.textContent).toContain("delta new");
  });

  it("preserves retry prompt across a same-turn message identity replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "retry identity prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start", { turn_id: "retry-turn", message_id: "retry-old" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "retry-turn", message_id: "retry-old" });
      gateway.instance?.emit("message.delta", { text: "new partial", turn_id: "retry-turn", message_id: "retry-new" });
      gateway.instance?.emit("error", { turn_id: "retry-turn", message_id: "retry-new", error: "replacement failed" });
    });

    expect(host.querySelector("button[aria-label='Retry send']")).toBeTruthy();
    expect(host.textContent).toContain("replacement failed");
  });

  it("retains text across tagged and untagged stream events", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "first " });
      gateway.instance?.emit("message.delta", { text: "tagged ", turn_id: "turn-1" });
      gateway.instance?.emit("message.delta", { text: "tail" });
      gateway.instance?.emit("message.complete");
    });

    expect(host.textContent).toContain("first tagged tail");
  });

  it("blocks a seq-less terminal between completion and a replacement start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "first prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "first answer" });
      gateway.instance?.emit("message.complete");
    });
    await act(async () => {
      setter?.call(textarea, "replacement prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.complete", { text: "stale completion" });
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "replacement answer" });
    });

    expect(host.textContent).not.toContain("stale completion");
    expect(host.textContent).toContain("replacement answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("recovers a submitted turn when delta arrives before an untagged start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "delta first prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.delta", { text: "delta first answer" });
      gateway.instance?.emit("message.complete", { text: "delta first answer" });
    });

    expect(host.textContent).toContain("delta first answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("keeps identical assistant answers as separate turns", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    const submitTurn = async (prompt: string) => {
      await act(async () => {
        setter?.call(textarea, prompt);
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      });
      await act(async () => {
        gateway.instance?.emit("message.start");
        gateway.instance?.emit("message.delta", { text: "same answer" });
        gateway.instance?.emit("message.complete");
      });
    };

    await submitTurn("first prompt");
    await submitTurn("second prompt");

    const messages = Array.from(host.querySelectorAll<HTMLElement>("[data-slot='transcript-message']"));
    expect(messages.map((message) => message.dataset.messageRole)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(messages).toHaveLength(4);
    expect(messages[1]?.textContent).toContain("same answer");
    expect(messages[3]?.textContent).toContain("same answer");
  });

  it("filters transcript rows from the message search toolbar and can clear the query", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const textareaSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      textareaSetter?.call(textarea, "searchable prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "needle answer" });
      gateway.instance?.emit("message.complete");
    });

    const search = host.querySelector<HTMLInputElement>("input[aria-label='Search message content']")!;
    const searchSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    await act(async () => {
      searchSetter?.call(search, "needle");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(host.querySelectorAll("[data-slot='transcript-row']")).toHaveLength(1);
    expect(host.querySelector("[data-slot='transcript-search']")?.textContent).toContain("1 match");
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Clear message search']")?.click());
    expect(host.querySelectorAll("[data-slot='transcript-row']").length).toBeGreaterThan(1);
  });

  it("branches the active session from the command palette using the durable branch id", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => window.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "k", ctrlKey: true })));
    const branch = host.querySelector<HTMLButtonElement>("button[aria-label='Branch current session']");
    expect(branch).toBeTruthy();
    await act(async () => branch?.click());
    expect(gateway.instance?.requests.some(({ method, params }) => method === "session.branch" && params.session_id === "session-1")).toBe(true);
  });

  it("confirms a durable edit before truncating and submits the row-id contract", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=stored-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const editButton = host.querySelector<HTMLButtonElement>("button[aria-label='Edit user message']");
    expect(editButton).toBeTruthy();
    await act(async () => editButton?.click());
    expect(document.body.querySelector("[role='dialog']")).toBeTruthy();
    expect(gateway.instance?.requests.some(({ method }) => method === "prompt.submit")).toBe(false);

    const editTextarea = document.body.querySelector<HTMLTextAreaElement>("textarea[aria-label='Edited user message']")!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(editTextarea, "revised first prompt");
      editTextarea.dispatchEvent(new Event("input", { bubbles: true }));
      document.body.querySelector<HTMLButtonElement>("button[data-confirm]")?.click();
    });
    const editRequest = gateway.instance?.requests.find(({ method }) => method === "prompt.submit");
    expect(editRequest?.params).toMatchObject({
      session_id: "runtime-1",
      text: "revised first prompt",
      truncate_before_row_id: 7,
      confirm_truncate: true,
      confirm_empty_truncate: true,
      rebind_survivor_row_ids: [7],
    });
    expect(document.body.querySelector("[role='dialog']")).toBeNull();
  });

  it("ignores a late edit success after New Chat starts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Edit user message']")?.click());
    const editTextarea = document.body.querySelector<HTMLTextAreaElement>("textarea[aria-label='Edited user message']")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    let releaseEdit!: () => void;
    let editStarted!: () => void;
    const editStartedPromise = new Promise<void>((resolve) => { editStarted = resolve; });
    gateway.instance!.submitBlock = new Promise<void>((resolve) => { releaseEdit = resolve; });
    gateway.instance!.onSubmitBlockConsumed = editStarted;
    await act(async () => {
      setter?.call(editTextarea, "stale edited prompt");
      editTextarea.dispatchEvent(new Event("input", { bubbles: true }));
      document.body.querySelector<HTMLButtonElement>("button[data-confirm]")?.click();
    });
    await editStartedPromise;
    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => {
      releaseEdit();
      await Promise.resolve();
    });

    expect(host.textContent).not.toContain("stale edited prompt");
    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
  });

  it("does not submit a durable edit while Stop is still pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Edit user message']")?.click());
    const editTextarea = document.body.querySelector<HTMLTextAreaElement>("textarea[aria-label='Edited user message']")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(editTextarea, "blocked edited prompt");
      editTextarea.dispatchEvent(new Event("input", { bubbles: true }));
      gateway.instance?.emit("message.start", { turn_id: "active-turn" }, "runtime-1");
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => document.body.querySelector<HTMLButtonElement>("button[data-confirm]")?.click());

    expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "blocked edited prompt")).toHaveLength(0);
    expect(host.textContent).toContain("Wait for Stop to finish before editing");
    await act(async () => releaseInterrupt());
  });

  it("supports edit-as-draft and rerunning the latest prompt", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "original prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "answer" });
      gateway.instance?.emit("message.complete");
    });

    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Edit user message']")?.click());
    expect(host.querySelector<HTMLTextAreaElement>("textarea")?.value).toBe("original prompt");
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Run assistant message again']")?.click());
    expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "original prompt")).toHaveLength(2);
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
      const composingEnter = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter", isComposing: false });
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

  it("preserves exact Thai Unicode from controlled browser input", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const pastedThai = "กำลังทดสอบ ั\u0e33";
    await act(async () => {
      // jsdom cannot perform the browser's native text-paste insertion; model
      // the resulting controlled textarea change and verify the RPC payload.
      const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      setter?.call(textarea, pastedThai);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
    });
    expect(gateway.instance?.requests.find(({ method }) => method === "prompt.submit")?.params.text).toBe(pastedThai);
  });

  it("stages image and file attachments with the gateway payloads", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:photo-preview") });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
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
    expect(host.querySelector("[data-slot='attachment-preview']")?.getAttribute("src")).toBe("blob:photo-preview");
    expect(host.querySelector("[data-slot='attachment']")?.textContent).toMatch(/\d+ B/);
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

  it("collapses completed tool activity into an expandable summary", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("tool.start", { tool_id: "done-1", name: "read_file" });
      gateway.instance?.emit("tool.complete", { tool_id: "done-1", name: "read_file", result: { ok: true } });
      gateway.instance?.emit("tool.start", { tool_id: "done-2", name: "search_files" });
      gateway.instance?.emit("tool.complete", { tool_id: "done-2", name: "search_files", result: { matches: 2 } });
    });
    const timeline = host.querySelector("[data-slot='tool-timeline']");
    const summary = timeline?.querySelector("[data-slot='completed-tool-activity']");
    expect(summary).not.toBeNull();
    expect(summary?.textContent).toContain("Background work complete");
    expect(summary?.textContent).toContain("2 steps");
    expect((summary as HTMLDetailsElement | null)?.open).toBe(false);
    await act(async () => summary?.querySelector<HTMLElement>("summary")?.click());
    expect((summary as HTMLDetailsElement | null)?.open).toBe(true);
    expect(summary?.querySelectorAll("[data-tool-state='complete']")).toHaveLength(2);
  });

  it("does not carry completed tool activity into a new assistant turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("tool.start", { tool_id: "old-tool", name: "old tool" });
      gateway.instance?.emit("tool.complete", { tool_id: "old-tool", name: "old tool" });
    });
    expect(host.querySelector("[data-slot='completed-tool-activity']")).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "new-tool-turn" }));
    expect(host.querySelector("[data-slot='completed-tool-activity']")).toBeNull();
  });

  it("places running tool activity and completed summaries in the transcript", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("tool.start", { tool_id: "tool-1", name: "first" });
      gateway.instance?.emit("tool.complete", { tool_id: "tool-1", name: "first", result: { ok: true } });
      gateway.instance?.emit("tool.start", { tool_id: "tool-2", name: "second" });
    });
    const transcript = host.querySelector("[data-testid='native-chat-transcript']");
    const timeline = host.querySelector("[data-testid='tool-timeline']");
    expect(timeline?.parentElement).toBe(transcript);
    expect(Array.from(timeline?.querySelectorAll("[data-tool-id]") ?? []).map((item) => item.getAttribute("data-tool-id"))).toEqual(["tool-2", "tool-1"]);
    expect(timeline?.querySelector("[data-tool-id='tool-1']")?.getAttribute("data-tool-state")).toBe("complete");
    expect(timeline?.querySelector("[data-tool-id='tool-2']")?.getAttribute("data-tool-state")).toBe("running");
    expect((timeline?.querySelector("[data-slot='completed-tool-activity']") as HTMLDetailsElement | null)?.open).toBe(false);
  });

  it("renders tool timing from the completion event", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("tool.start", { tool_id: "timed-tool", name: "terminal" });
      gateway.instance?.emit("tool.complete", { tool_id: "timed-tool", name: "terminal", elapsed_ms: 1500 });
    });
    expect(host.querySelector("[data-tool-id='timed-tool']")?.textContent).toContain("1.5s");
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
    expect(gateway.instance?.requests.at(-1)).toEqual({ method: "approval.respond", params: { choice: "once", request_id: "approval-1", session_id: "session-1", profile: "thai-profile" } });
    await act(async () => gateway.instance?.emit("clarify.request", { request_id: "clarify-1", question: "Which?", choices: ["A"] }));
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='A']")?.click());
    expect(gateway.instance?.requests.at(-1)).toEqual({ method: "clarify.respond", params: { answer: "A", profile: "thai-profile", request_id: "clarify-1", session_id: "session-1" } });
  });

  it("mounts the approval dialog and buttons after transcript content and tool activity so auto-scroll keeps them in view", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const textareaSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      textareaSetter?.call(textarea, "run dangerous task");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "Executing requested command..." });
      gateway.instance?.emit("tool.start", { tool_id: "tool-test-1", name: "terminal", context: "running" });
      gateway.instance?.emit("approval.request", {
        request_id: "approval-order-test",
        command: "rm -rf /tmp/test",
        choices: ["once", "deny"],
      });
    });

    const transcript = host.querySelector<HTMLElement>("[data-slot='transcript']");
    expect(transcript).not.toBeNull();

    const transcriptRows = host.querySelectorAll<HTMLElement>("[data-slot='transcript-row']");
    expect(transcriptRows.length).toBeGreaterThan(0);
    const lastTranscriptRow = transcriptRows[transcriptRows.length - 1];

    const toolTimeline = host.querySelector<HTMLElement>("[data-slot='tool-timeline']");
    expect(toolTimeline).not.toBeNull();

    const approvalDialog = host.querySelector<HTMLElement>("[role='dialog']");
    expect(approvalDialog).not.toBeNull();
    expect(approvalDialog?.getAttribute("aria-label")).toBe("Approval required");

    const approvalButton = host.querySelector<HTMLButtonElement>("button[data-choice='once']");
    expect(approvalButton).not.toBeNull();

    const spacers = transcript!.querySelectorAll<HTMLElement>("[data-slot='transcript-virtual-spacer']");
    const bottomSpacer = spacers[spacers.length - 1];
    expect(bottomSpacer).not.toBeNull();

    // Approval dialog and buttons must be mounted after the transcript content so auto-scroll keeps them in view
    expect(Boolean(lastTranscriptRow.compareDocumentPosition(approvalDialog!) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
    expect(Boolean(toolTimeline!.compareDocumentPosition(approvalDialog!) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
    expect(Boolean(bottomSpacer.compareDocumentPosition(approvalDialog!) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
    expect(Boolean(lastTranscriptRow.compareDocumentPosition(approvalButton!) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
  });

  it("queues distinct approvals and promotes the next server request after resolution", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    await act(async () => {
      gateway.instance?.emit("approval.request", { request_id: "approval-first", command: "rm first-file", choices: ["once", "deny"] });
      gateway.instance?.emit("approval.request", { request_id: "approval-second", command: "rm second-file", choices: ["once", "deny"] });
    });
    expect(host.textContent).toContain("rm first-file");
    expect(host.textContent).not.toContain("rm second-file");
    expect(host.querySelector("button[aria-label='Stop']")).not.toBeNull();
    expect(host.querySelector("button[aria-label='Queue message']")).not.toBeNull();
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    await vi.waitFor(() => expect(host.textContent).toContain("rm second-file"));
    expect(host.textContent).not.toContain("rm first-file");
    expect(gateway.instance?.requests.find(({ method, params }) => method === "approval.respond" && params.request_id === "approval-first")?.params).toEqual({ choice: "once", request_id: "approval-first", session_id: "session-1", profile: "thai-profile" });
  });

  it("honors approval capabilities and makes malformed explicit choices deny-only", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-capabilities", command: "capability-test", choices: ["session", "always", "deny"], allow_session: true, allow_permanent: false }));
    expect(host.querySelector("button[data-choice='session']")).not.toBeNull();
    expect(host.querySelector("button[data-choice='always']")).toBeNull();
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='session']")?.click());
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-malformed", command: "malformed-test", choices: [], allow_session: "yes", allow_permanent: 1 }));
    await vi.waitFor(() => expect(host.textContent).toContain("malformed-test"));
    expect(host.querySelector("button[data-choice='once']")).toBeNull();
    expect(host.querySelector("button[data-choice='session']")).toBeNull();
    expect(host.querySelector("button[data-choice='always']")).toBeNull();
    expect(host.querySelector("button[data-choice='deny']")).not.toBeNull();
  });

  it("does not render an approval card without a non-empty request id", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("approval.request", { request_id: " ", command: "rm file", choices: ["once"] }));
    expect(host.querySelector("[role='dialog']")).toBeNull();
  });

  it("keeps a native approval visible when the backend reports resolved zero", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.approvalResponse = { resolved: 0 };
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-stale", command: "rm stale", choices: ["once"] }));
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    expect(host.textContent).toContain("rm stale");
    expect(host.textContent).toContain("stale, expired, or already resolved");
  });

  it("ignores a late approval response after New Chat starts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-old", command: "rm old-file", choices: ["once"] }));
    let releaseApproval!: () => void;
    let approvalStarted!: () => void;
    const approvalStartedPromise = new Promise<void>((resolve) => { approvalStarted = resolve; });
    gateway.instance!.approvalBlock = new Promise<void>((resolve) => { releaseApproval = resolve; });
    gateway.instance!.onApprovalBlockConsumed = approvalStarted;
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    await approvalStartedPromise;

    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => {
      releaseApproval();
      await Promise.resolve();
    });

    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
    expect(host.textContent).not.toContain("rm old-file");
  });

  it("deduplicates seq-less replay events by event id", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { event_id: "start-1" });
      gateway.instance?.emit("message.delta", { text: "once", event_id: "delta-1" });
      gateway.instance?.emit("message.delta", { text: "once", event_id: "delta-1" });
    });

    expect(host.textContent).toContain("once");
    expect(host.textContent).not.toContain("onceonce");
  });

  it("does not let a snapshot without pending fields erase a live approval", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-live", command: "rm file", choices: ["once"] }));
    expect(host.textContent).toContain("rm file");

    gateway.instance!.snapshot = { messages: [{ id: 9, role: "assistant", text: "history" }] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("rm file");
    expect(host.querySelector("[aria-label='Approval required']")).toBeTruthy();
  });

  it("preserves live content when an idle snapshot omits transcript fields", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "live-message" });
      gateway.instance?.emit("message.delta", { text: "live answer", message_id: "live-message" });
    });
    gateway.instance!.snapshot = { session_id: "session-1", running: false, status: "idle" };
    gateway.instance!.omitResumeMessages = true;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("live answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("adopts a new runtime id returned by reconnect resume fallback", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-reconnected", stored_session_id: "stored-1", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));

    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "reconnected-turn" }, "runtime-reconnected");
      gateway.instance?.emit("message.delta", { text: "accepted after runtime adoption", turn_id: "reconnected-turn" }, "runtime-reconnected");
    });
    expect(host.textContent).toContain("accepted after runtime adoption");
  });

  it("keeps delta-only continuity when reconnect adopts a new runtime", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "adopt-turn", message_id: "adopt-message" });
      gateway.instance?.emit("message.delta", { text: "before reconnect", turn_id: "adopt-turn", message_id: "adopt-message" });
    });
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-adopted", stored_session_id: "stored-1", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => gateway.instance?.emit("message.delta", { text: " after reconnect", turn_id: "adopt-turn", message_id: "adopt-message" }, "runtime-adopted"));

    expect(host.textContent).toContain("before reconnect after reconnect");
  });

  it("does not duplicate a same-turn empty inflight row without a user anchor", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "atomic-empty-turn", message_id: "atomic-empty-message" });
      gateway.instance?.emit("message.delta", { turn_id: "atomic-empty-turn", message_id: "atomic-empty-message", text: "live atomic partial" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      messages: [],
      running: true,
      inflight: {
        assistant: "",
        streaming: true,
        turn_id: "atomic-empty-turn",
        message_id: "atomic-empty-message",
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(1);
    expect(assistants[0]?.textContent).toContain("live atomic partial");
  });

  it("does not split an active message-id-only stream on an unbound start replay", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "active-only-message" });
      gateway.instance?.emit("message.delta", { message_id: "active-only-message", text: "active-only prefix" });
      gateway.instance?.emit("message.start");
    });
    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(1);
    expect(assistants[0]?.textContent).toContain("active-only prefix");
  });

  it("merges an identity-bearing empty inflight error onto durable history", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-empty-error",
      stored_session_id: "empty-error-durable",
      running: false,
      messages: [
        { id: "durable-error-user", role: "user", text: "empty error prompt" },
        { message_id: "empty-error-message", role: "assistant", text: "durable partial answer" },
      ],
      inflight: {
        user: "empty error prompt",
        assistant: "",
        status: "error",
        error: "empty assistant failure",
        turn_id: "empty-error-turn",
        message_id: "empty-error-message",
        streaming: false,
      },
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=empty-error-durable"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("empty assistant failure"));
    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(1);
    expect(assistants[0]?.textContent).toContain("durable partial answer");
    expect(assistants[0]?.getAttribute("data-message-error")).toBe("true");
  });

  it("preserves an inflight error when a same-turn live assistant is present", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "live-error-turn", message_id: "live-error-message" });
      gateway.instance?.emit("message.delta", { turn_id: "live-error-turn", message_id: "live-error-message", text: "live partial" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: false,
      messages: [],
      inflight: {
        user: "live error prompt",
        assistant: "live partial",
        streaming: false,
        status: "error",
        error: "provider failure",
        turn_id: "live-error-turn",
        message_id: "live-error-message",
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    const matching = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .filter((message) => message.textContent?.includes("live partial"));
    expect(matching).toHaveLength(1);
    expect(matching[0]?.getAttribute("data-message-error")).toBe("true");
    expect(matching[0]?.textContent).toContain("provider failure");
  });

  it("preserves the longer live text when same-turn inflight snapshot is stale", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "same-inflight-turn", message_id: "same-inflight-message" });
      gateway.instance?.emit("message.delta", { text: "long live partial", turn_id: "same-inflight-turn", message_id: "same-inflight-message" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      inflight: { user: "same prompt", assistant: "long live", streaming: true, turn_id: "same-inflight-turn", message_id: "same-inflight-message" },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain("long live partial");
  });
  it("commits and retires a local turn before adopting a different inflight turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "local-turn", message_id: "local-message" });
      gateway.instance?.emit("message.delta", { text: "local partial", turn_id: "local-turn", message_id: "local-message" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      messages: [],
      running: true,
      inflight: {
        user: "different prompt",
        assistant: "different inflight answer",
        streaming: true,
        turn_id: "different-turn",
        message_id: "different-message",
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("local partial");
    expect(host.textContent).toContain("different inflight answer");
    await act(async () => gateway.instance?.emit("message.complete", { text: "local resurrection", turn_id: "local-turn", message_id: "local-message" }));
    expect(host.textContent).not.toContain("local resurrection");
  });

  it("treats a same-turn new-message snapshot as a replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "shared-turn", message_id: "snapshot-old-message" });
      gateway.instance?.emit("message.delta", { text: "snapshot old partial", turn_id: "shared-turn", message_id: "snapshot-old-message" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      messages: [],
      running: true,
      inflight: { user: "snapshot replacement", assistant: "snapshot new answer", streaming: true, turn_id: "shared-turn", message_id: "snapshot-new-message" },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("snapshot old partial");
    expect(host.textContent).toContain("snapshot new answer");
  });

  it("does not apply a running snapshot after the turn completed during resync", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "resync-old-turn", message_id: "resync-old-message" });
      gateway.instance?.emit("message.delta", { text: "old live answer", turn_id: "resync-old-turn", message_id: "resync-old-message" });
    });
    let releaseActivate!: () => void;
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.activateBlock).toBe(null));
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      inflight: {
        user: "resync old prompt",
        assistant: "resurrected answer",
        streaming: true,
        turn_id: "resync-old-turn",
        message_id: "resync-old-message",
      },
    };
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "resync-old-turn", message_id: "resync-old-message" }));
    await act(async () => { releaseActivate(); await Promise.resolve(); });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).not.toContain("resurrected answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.querySelector("[data-slot='chat-resync']")).toBeNull();
  });

  it("retries resync after a concurrent submit invalidates the snapshot", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    let releaseActivate!: () => void;
    let activateStarted!: () => void;
    const activateStartedPromise = new Promise<void>((resolve) => { activateStarted = resolve; });
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = activateStarted;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await activateStartedPromise;
    const initialActivates = gateway.instance!.requests.filter(({ method }) => method === "session.activate").length;

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "resync replacement prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
    });
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], running: true, inflight: { user: "resync replacement prompt", assistant: "replacement inflight", streaming: true, turn_id: "replacement-turn", message_id: "replacement-message" } };
    await act(async () => { releaseActivate(); await Promise.resolve(); });

    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method }) => method === "session.activate").length).toBeGreaterThan(initialActivates));
  });

  it("does not surface a stale resync failure after the barrier is invalidated", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    let releaseActivate!: () => void;
    let activateStarted!: () => void;
    const activateStartedPromise = new Promise<void>((resolve) => { activateStarted = resolve; });
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.activateErrorOnce = true;
    gateway.instance!.resumeErrorOnce = true;
    gateway.instance!.onActivateBlockConsumed = activateStarted;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await activateStartedPromise;
    const initialActivates = gateway.instance!.requests.filter(({ method }) => method === "session.activate").length;

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "operation wins over resync");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      releaseActivate();
      await Promise.resolve();
    });

    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method }) => method === "session.activate").length).toBeGreaterThan(initialActivates));
    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
    expect(host.textContent).not.toContain("Resync failed");
  });

  it("does not retry resync while Stop is still pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "resync-stop-turn", message_id: "resync-stop-message" }));
    await act(async () => gateway.instance?.emit("message.delta", { text: "active before reconnect", turn_id: "resync-stop-turn", message_id: "resync-stop-message" }));
    let releaseActivate!: () => void;
    const activateStarted = new Promise<void>((resolve) => {
      gateway.instance!.activateBlock = new Promise<void>((resolveActivate) => { releaseActivate = resolveActivate; });
      gateway.instance!.onActivateBlockConsumed = resolve;
    });
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await activateStarted;
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.interrupt")).toBe(true));
    const resyncCount = gateway.instance?.requests.filter(({ method }) => method === "session.activate" || method === "session.resume").length ?? 0;
    await act(async () => { releaseActivate(); await Promise.resolve(); });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(gateway.instance?.requests.filter(({ method }) => method === "session.activate" || method === "session.resume").length).toBe(resyncCount);
    expect(host.querySelector("[data-slot='chat-notices']")?.getAttribute("data-resync-state")).not.toBe("syncing");
    await act(async () => { releaseInterrupt(); await Promise.resolve(); });
  });
  it("resets the reconnect guard when New Chat replaces an in-flight resync", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    let releaseActivate!: () => void;
    let activateStarted!: () => void;
    const activateStartedPromise = new Promise<void>((resolve) => { activateStarted = resolve; });
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = activateStarted;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await activateStartedPromise;

    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method }) => method === "session.create").length).toBeGreaterThanOrEqual(2));
    const activateCount = gateway.instance?.requests.filter(({ method }) => method === "session.activate").length ?? 0;
    await act(async () => { releaseActivate(); await Promise.resolve(); });

    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method }) => method === "session.activate").length).toBeGreaterThan(activateCount));
  });

  it("deduplicates an event replayed under a replacement runtime id", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start", { event_id: "runtime-replay-start" }));
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-reconnected", stored_session_id: "stored-1", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => gateway.instance?.emit("message.start", { event_id: "runtime-replay-start" }, "runtime-reconnected"));

    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
  });

  it("keeps a retired Stop target effective after runtime adoption", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "stop-runtime-old", message_id: "stop-runtime-old-message" });
      gateway.instance?.emit("message.delta", { text: "old stop stream", turn_id: "stop-runtime-old", message_id: "stop-runtime-old-message" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-reconnected", stored_session_id: "stored-1", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "stop-runtime-new", message_id: "stop-runtime-new-message" }, "runtime-reconnected");
      gateway.instance?.emit("message.delta", { text: "new stop stream", turn_id: "stop-runtime-new", message_id: "stop-runtime-new-message" }, "runtime-reconnected");
      gateway.instance?.emit("message.complete", { text: "stale stop terminal" }, "runtime-reconnected");
    });

    expect(host.textContent).toContain("new stop stream");
    expect(host.textContent).not.toContain("stale stop terminal");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("rejects a completed message-id replay after runtime adoption", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "completed-message" });
      gateway.instance?.emit("message.delta", { text: "completed answer", message_id: "completed-message" });
      gateway.instance?.emit("message.complete", { message_id: "completed-message" });
    });
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-reconnected", stored_session_id: "stored-1", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => gateway.instance?.emit("message.start", { message_id: "completed-message" }, "runtime-reconnected"));

    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
  });

  it("rekeys identity-less inflight fallback state with a canonical scope change", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-scope-old",
      session_key: "canonical-k",
      stored_session_id: "alias-k2",
      running: true,
      inflight: { user: "scope prompt", assistant: "before scope", streaming: true },
      messages: [],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=canonical-k"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("before scope"));
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = {
      session_id: "runtime-scope-new",
      session_key: "alias-k2",
      stored_session_id: undefined,
      running: true,
      inflight: { user: "scope prompt", assistant: "before scope continued", streaming: true },
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(1);
    expect(host.textContent).toContain("before scope continued");
  });

  it("keeps identical identity-less inflight data separate after a new explicit turn", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-fallback-old",
      stored_session_id: "fallback-durable",
      running: true,
      inflight: { assistant: "identical fallback answer", streaming: true },
      messages: [],
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=fallback-durable"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("identical fallback answer"));
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "new-explicit-turn", message_id: "new-explicit-message" }, "runtime-fallback-old"));
    await act(async () => gateway.instance?.emit("message.delta", { turn_id: "new-explicit-turn", message_id: "new-explicit-message", text: "new explicit answer" }, "runtime-fallback-old"));
    gateway.instance!.activateError = true;
    gateway.instance!.snapshot = { session_id: "runtime-fallback-new", stored_session_id: "fallback-durable", running: true, inflight: { assistant: "identical fallback answer", streaming: true }, messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect((host.textContent?.match(/identical fallback answer/g) ?? []).length).toBe(2);
  });

  it("keeps one inflight row with stable started_at identity as its text grows", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], inflight: { assistant: "part", streaming: true, started_at: 1700000000 } };
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); await Promise.resolve(); });
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], inflight: { assistant: "partial answer", streaming: true, started_at: 1700000000 } };
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); await Promise.resolve(); });
    const assistantRows = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistantRows).toHaveLength(1);
    expect(assistantRows[0]?.textContent).toContain("partial answer");
  });

  it("does not duplicate an assistant-only inflight snapshot with stable started_at identity", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], running: true, inflight: { assistant: "stable identity-less answer", streaming: true, started_at: 1700000000 } };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect((host.textContent?.match(/stable identity-less answer/g) ?? []).length).toBe(1);
  });
  it("keeps distinct identity-less inflight snapshots as separate turns", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = {
      session_id: "session-1",
      messages: [],
      running: true,
      inflight: { user: "first inflight prompt", assistant: "first inflight answer", streaming: true },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      messages: [],
      running: true,
      inflight: { user: "second inflight prompt", assistant: "second inflight answer", streaming: true },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("first inflight answer");
    expect(host.textContent).toContain("second inflight answer");
  });

  it("does not treat an empty identity-less snapshot as continuation of a non-empty stream", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], running: true, inflight: { assistant: "non-empty partial", streaming: true } };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], running: true, inflight: { assistant: "", streaming: true } };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(2);
  });

  it("keeps distinct same-prompt identity-less inflight snapshots separate", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], running: true, inflight: { user: "same prompt", assistant: "first answer", streaming: true } };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = { session_id: "session-1", messages: [], running: true, inflight: { user: "same prompt", assistant: "second answer", streaming: true } };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(2);
    expect(host.textContent).toContain("first answer");
    expect(host.textContent).toContain("second answer");
  });

  it("clears a retired turn fence when resync adopts a same-turn inflight replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "inflight-replacement-turn", message_id: "inflight-old-message" });
      gateway.instance?.emit("message.delta", { turn_id: "inflight-replacement-turn", message_id: "inflight-old-message", text: "old inflight answer" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      messages: [],
      running: true,
      inflight: {
        assistant: "new inflight answer",
        streaming: true,
        turn_id: "inflight-replacement-turn",
        message_id: "inflight-new-message",
      },
    };
    await act(async () => { gateway.instance?.stateHandler?.("closed"); gateway.instance?.stateHandler?.("open"); await Promise.resolve(); });
    await vi.waitFor(() => expect(host.textContent).toContain("new inflight answer"));
    await act(async () => gateway.instance?.emit("message.delta", {
      turn_id: "inflight-replacement-turn",
      message_id: "inflight-new-message",
      text: " continued",
    }));

    expect(host.textContent).toContain("new inflight answer continued");
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).not.toBeNull();
  });

  it("does not replay lower-sequence stale events after a resync watermark", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "watermark-old-turn" }, "session-1", 10);
      gateway.instance?.emit("message.delta", { turn_id: "watermark-old-turn", text: "watermark old answer" }, "session-1", 11);
    });
    let releaseActivate!: () => void;
    let activateStarted!: () => void;
    const activateStartedPromise = new Promise<void>((resolve) => { activateStarted = resolve; });
    gateway.instance!.snapshot = { session_id: "session-1", messages: [] };
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = activateStarted;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await activateStartedPromise;
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "stale-buffered-turn" }, "session-1", 5);
      gateway.instance?.emit("message.delta", { turn_id: "stale-buffered-turn", text: "stale buffered answer" }, "session-1", 6);
      releaseActivate();
      await Promise.resolve();
    });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).not.toContain("stale buffered answer");
  });

  it("accepts a restarted sequence after reconnecting the same runtime", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "sequence-old" }, "session-1", 10);
      gateway.instance?.emit("message.delta", { turn_id: "sequence-old", text: "before sequence reset" }, "session-1", 11);
    });
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "session-1", stored_session_id: "stored-1", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "sequence-new" }, "session-1", 1);
      gateway.instance?.emit("message.delta", { turn_id: "sequence-new", text: "after sequence reset" }, "session-1", 2);
    });
    expect(host.textContent).toContain("after sequence reset");
  });

  it("accepts a replacement runtime whose per-runtime sequence restarts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-runtime-turn" }, "session-1", 1);
      gateway.instance?.emit("message.delta", { text: "old runtime", turn_id: "old-runtime-turn" }, "session-1", 2);
    });
    gateway.instance!.activateError = true;
    gateway.instance!.resumeResponse = { session_id: "runtime-reconnected", stored_session_id: "stored-1", messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.resume")).toBe(true));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-runtime-turn", message_id: "new-runtime-message" }, "runtime-reconnected", 1);
      gateway.instance?.emit("message.delta", { text: "new runtime accepted", turn_id: "new-runtime-turn", message_id: "new-runtime-message" }, "runtime-reconnected", 2);
    });

    expect(host.textContent).toContain("new runtime accepted");
  });

  it("does not apply a stale idle resync snapshot after a new prompt starts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    let releaseActivate!: () => void;
    let activateStarted!: () => void;
    const activateStartedPromise = new Promise<void>((resolve) => { activateStarted = resolve; });
    gateway.instance!.snapshot = { session_id: "session-1", running: false, messages: [] };
    gateway.instance!.activateBlock = new Promise<void>((resolve) => { releaseActivate = resolve; });
    gateway.instance!.onActivateBlockConsumed = activateStarted;
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await activateStartedPromise;

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "new prompt during resync");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[type='submit']")?.click());
    await act(async () => {
      releaseActivate();
      await Promise.resolve();
    });
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn" });
      gateway.instance?.emit("message.delta", { text: "new turn survived stale snapshot", turn_id: "new-turn" });
    });

    expect(host.textContent).toContain("new prompt during resync");
    expect(host.textContent).toContain("new turn survived stale snapshot");
    expect(host.querySelector("[data-slot='chat-status']")?.textContent).toContain("Working");
  });

  it("clears submit state when a concurrent approval operation invalidates its result", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    let releaseSubmit!: () => void;
    let submitStarted!: () => void;
    const submitStartedPromise = new Promise<void>((resolve) => { submitStarted = resolve; });
    gateway.instance!.submitBlock = new Promise<void>((resolve) => { releaseSubmit = resolve; });
    gateway.instance!.onSubmitBlockConsumed = submitStarted;
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "blocked prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });
    await submitStartedPromise;

    await act(async () => gateway.instance?.emit("approval.request", { request_id: "concurrent-approval", choices: ["once"] }));
    let rejectApproval!: (reason: Error) => void;
    let approvalStarted!: () => void;
    const approvalStartedPromise = new Promise<void>((resolve) => { approvalStarted = resolve; });
    gateway.instance!.approvalBlock = new Promise<void>((_, reject) => { rejectApproval = reject; });
    gateway.instance!.onApprovalBlockConsumed = approvalStarted;
    await vi.waitFor(() => expect(host.querySelector("button[data-choice='once']")).toBeTruthy());
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    await approvalStartedPromise;
    await act(async () => { rejectApproval(new Error("approval failed")); await Promise.resolve(); });
    await act(async () => { releaseSubmit(); await Promise.resolve(); });

    const followUpSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    const followUp = host.querySelector<HTMLTextAreaElement>("textarea")!;
    await act(async () => {
      followUpSetter?.call(followUp, "follow up");
      followUp.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await vi.waitFor(() => expect(host.querySelector<HTMLButtonElement>("button[type='submit']")?.disabled).toBe(false));
  });

  it("keeps a newer approval request when an older response resolves", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    let releaseApproval!: () => void;
    let approvalStarted!: () => void;
    const approvalStartedPromise = new Promise<void>((resolve) => { approvalStarted = resolve; });
    gateway.instance!.approvalBlock = new Promise<void>((resolve) => { releaseApproval = resolve; });
    gateway.instance!.onApprovalBlockConsumed = approvalStarted;
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-old", choices: ["once"] }));
    await act(async () => host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.click());
    await approvalStartedPromise;
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "approval-new", choices: ["once"] }));
    await act(async () => { releaseApproval(); await Promise.resolve(); });

    expect(host.querySelector("[aria-label='Approval required']")?.textContent).toContain("once");
    expect(host.textContent).not.toContain("approval failed");
  });

  it("ignores a late Stop response after New Chat changes session generation", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "stop-old" });
      gateway.instance?.emit("message.delta", { text: "old stream", turn_id: "stop-old" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { releaseInterrupt(); await Promise.resolve(); });

    expect(host.querySelector("[data-slot='chat-status']")?.textContent).not.toContain("Stopped");
    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
  });

  it("does not restore a retired Stop target after a replacement untagged stream starts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old stream", message_id: "old-message" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new stream", message_id: "new-message" });
    });
    await act(async () => { releaseInterrupt(); await Promise.resolve(); });
    await act(async () => gateway.instance?.emit("message.complete", { text: "new complete", message_id: "new-message" }));

    expect(host.textContent).toContain("new complete");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("clears a retired Stop target for a distinct untagged inflight reconnect turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old stream", message_id: "old-message" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      messages: [],
      inflight: { user: "new prompt", assistant: "new partial", streaming: true, started_at: Date.now() / 1000 + 1 },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("message.complete", { text: "new final" }));

    expect(host.textContent).toContain("new final");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("does not restore a retired Stop target over a tagged replacement turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old stream", message_id: "old-message" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new stream", turn_id: "new-turn", message_id: "new-message" });
    });
    await act(async () => { releaseInterrupt(); await Promise.resolve(); });
    await act(async () => gateway.instance?.emit("message.complete", { text: "new final", message_id: "new-message" }));

    expect(host.textContent).toContain("new final");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("retires a tagged turn when session.info reports idle", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "retired-turn" });
      gateway.instance?.emit("message.delta", { text: "retired answer", turn_id: "retired-turn" });
      gateway.instance?.emit("session.info", { running: false, turn_id: "retired-turn" });
      gateway.instance?.emit("message.start", { turn_id: "retired-turn" });
      gateway.instance?.emit("message.delta", { text: "late resurrection", turn_id: "retired-turn" });
    });

    expect(host.textContent).toContain("retired answer");
    expect(host.textContent).not.toContain("late resurrection");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("stops a streaming session and deduplicates replayed deltas", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => { gateway.instance?.emit("message.start", undefined); gateway.instance?.emit("message.delta", { text: "one", seq: 2 }); gateway.instance?.emit("message.delta", { text: "one", seq: 2 }); });
    expect(host.textContent).toContain("one");
    expect(host.textContent).not.toContain("oneone");
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    expect(gateway.instance?.requests.at(-1)).toEqual({ method: "session.interrupt", params: { session_id: "session-1" } });
  });

  it("allows a subsequent untagged turn to complete after Stop settles", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn", message_id: "old-message" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "untagged answer" });
      gateway.instance?.emit("message.complete", { text: "untagged answer" });
    });

    expect(host.textContent).toContain("untagged answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("accepts a longer untagged replacement after Stop", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "prefix-stop-old", message_id: "prefix-stop-old-message" });
      gateway.instance?.emit("message.delta", { text: "stopped prefix", turn_id: "prefix-stop-old", message_id: "prefix-stop-old-message" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "stopped prefix plus legitimate replacement" });
    });

    expect(host.textContent).toContain("stopped prefix plus legitimate replacement");
    expect(host.querySelector("[data-message-role='assistant'][data-message-streaming='true']")).not.toBeNull();
  });

  it("blocks a stale seq-less terminal after a replacement starts after Stop settles", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn", message_id: "old-message" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { text: "stale old terminal" });
    });

    expect(host.textContent).toContain("new answer");
    expect(host.textContent).not.toContain("stale old terminal");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("clears prior tool and interaction state on an explicit replacement", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "control-old-turn", message_id: "control-old-message" });
      gateway.instance?.emit("message.delta", { turn_id: "control-old-turn", message_id: "control-old-message", text: "old control stream" });
    });
    await act(async () => gateway.instance?.emit("tool.start", { tool_id: "old-tool", name: "old tool", turn_id: "control-old-turn", message_id: "control-old-message" }));
    await act(async () => gateway.instance?.emit("approval.request", { request_id: "old-approval", command: "old command", choices: ["once", "deny"], turn_id: "control-old-turn", message_id: "control-old-message" }));
    await act(async () => gateway.instance?.emit("clarify.request", { request_id: "old-clarify", question: "old question", choices: ["A"], turn_id: "control-old-turn", message_id: "control-old-message" }));
    await vi.waitFor(() => expect(host.textContent).toContain("old command"));
    expect(host.textContent).toContain("old question");
    expect(host.querySelector("[data-slot='tool-timeline']")).not.toBeNull();
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "control-new-turn", message_id: "control-new-message" }));
    expect(host.textContent).not.toContain("old command");
    expect(host.textContent).not.toContain("old question");
    expect(host.querySelector("[data-slot='tool-timeline']")).toBeNull();
  });

  it("accepts a distinct replacement message terminal while Stop is still pending", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn", message_id: "old-message" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new stream", turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { text: "new complete", message_id: "new-message" });
    });

    expect(host.textContent).toContain("new complete");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    await act(async () => { releaseInterrupt(); await Promise.resolve(); });
  });

  it("blocks a stale seq-less terminal after a replacement prompt is submitted", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn", message_id: "old-message" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "new prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
      gateway.instance?.emit("message.start", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { text: "stale old terminal" });
    });

    expect(host.textContent).toContain("new answer");
    expect(host.textContent).not.toContain("stale old terminal");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("uses the accumulated stream text when complete only carries a shorter prefix", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "full answer" });
      gateway.instance?.emit("message.complete", { text: "full" });
    });
    expect(host.textContent).toContain("full answer");
  });

  it("clears local turn state when Stop resolves without a terminal event", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start"));
    await act(async () => gateway.instance?.emit("message.delta", { text: "partial answer" }));
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();

    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());

    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.querySelector("[data-slot='chat-status']")?.textContent).toContain("Ready");
    expect(host.textContent).toContain("partial answer");
  });

  it("does not let a late Stop response clear a newer turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { stop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });

    await act(async () => {
      gateway.instance?.emit("message.complete", { text: "old answer", turn_id: "old-turn" });
      gateway.instance?.emit("message.start", { turn_id: "new-turn" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn" });
    });
    await act(async () => {
      releaseInterrupt();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(host.textContent).toContain("new answer");
    expect(host.querySelector("[data-slot='turn-activity']")?.textContent).toContain("Thinking");
  });

  it("ignores a stale submit rejection after New Chat starts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });

    let rejectOld!: (reason?: unknown) => void;
    let submitStarted!: () => void;
    const submitStartedPromise = new Promise<void>((resolve) => { submitStarted = resolve; });
    gateway.instance!.submitBlock = new Promise<void>((_, reject) => { rejectOld = reject; });
    gateway.instance!.onSubmitBlockConsumed = submitStarted;

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "old prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });
    await submitStartedPromise;

    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => {
      rejectOld(new Error("old submit failed"));
      await Promise.resolve();
    });

    expect(host.textContent).not.toContain("old submit failed");
    expect(host.querySelector("[data-slot='chat-error']")).toBeNull();
  });

  it("does not let a late Stop response cross a New chat session boundary", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("message.start", undefined, "runtime-1"));
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { stop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });

    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "new session answer" });
    });
    await act(async () => {
      releaseInterrupt();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(host.textContent).toContain("new session answer");
    expect(host.querySelector("[data-slot='turn-activity']")?.textContent).toContain("Thinking");
  });

  it("does not drop an unbound session error after a turn completes", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.complete");
      gateway.instance?.emit("error", { message: "session fault" });
    });

    expect(host.textContent).toContain("session fault");
  });

  it("does not replace an active tagged stream with an untagged duplicate start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "active-turn" });
      gateway.instance?.emit("message.delta", { text: "active answer", turn_id: "active-turn" });
      gateway.instance?.emit("message.start");
    });

    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
    expect(host.textContent).toContain("active answer");
  });

  it("commits a terminal event received while a Stop request later fails", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn" });
      gateway.instance?.emit("message.delta", { text: "partial", turn_id: "old-turn" });
    });
    let rejectInterrupt!: (reason?: unknown) => void;
    gateway.instance!.interrupt = new Promise<void>((_, reject) => { rejectInterrupt = reject; });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { stop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });

    await act(async () => gateway.instance?.emit("message.complete", { text: "final answer", turn_id: "old-turn" }));
    await act(async () => {
      rejectInterrupt(new Error("interrupt failed"));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(host.textContent).toContain("final answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("cleans tools and pending controls when a terminal arrives during Stop", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn" });
      gateway.instance?.emit("tool.start", { tool_id: "running-tool", name: "terminal", turn_id: "old-turn" });
      gateway.instance?.emit("approval.request", { request_id: "approval-old", command: "rm file", choices: ["once"], turn_id: "old-turn" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => gateway.instance?.emit("message.complete", { text: "stopped", turn_id: "old-turn" }));

    expect(host.querySelector("[data-tool-id='running-tool']")?.getAttribute("data-tool-state")).toBe("complete");
    expect(host.querySelector("[aria-label='Approval required']")).toBeNull();
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    await act(async () => releaseInterrupt());
  });

  it("surfaces message.complete status errors as recoverable failures", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "provider prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
      gateway.instance?.emit("message.start", { turn_id: "error-turn" });
      gateway.instance?.emit("message.delta", { text: "partial", turn_id: "error-turn" });
      gateway.instance?.emit("message.complete", { text: "provider failed", status: "error", turn_id: "error-turn" });
    });

    expect(host.querySelector("[data-slot='chat-error']")?.textContent).toContain("provider failed");
    expect(host.querySelector("button[aria-label='Retry send']")).toBeTruthy();
    expect(host.querySelector("[data-slot='chat-status']")?.textContent).toContain("Error");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("attaches a terminal error to the durable row after live reconciliation", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "durable error prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start", { turn_id: "error-turn" });
      gateway.instance?.emit("message.delta", { text: "partial answer", turn_id: "error-turn" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      messages: [
        { row_id: 20, id: 20, role: "user", text: "durable error prompt" },
        { row_id: 21, id: 21, role: "assistant", text: "partial answer" },
      ],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("message.complete", { status: "error", error: "provider failure", turn_id: "error-turn" }));

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants).toHaveLength(1);
    expect(assistants[0]?.getAttribute("data-message-error")).toBe("true");
    expect(assistants[0]?.textContent).toContain("provider failure");
  });

  it("does not duplicate a recovered inflight error when a late complete arrives", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      session_id: "runtime-1",
      running: false,
      messages: [],
      inflight: {
        user: "failed prompt",
        assistant: "partial answer",
        status: "error",
        error: "provider failure",
        turn_id: "failed-turn",
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("message.complete", { text: "partial answer", status: "error", error: "provider failure" }, "runtime-1"));

    const matching = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .filter((message) => message.textContent?.includes("partial answer"));
    expect(matching).toHaveLength(1);
    expect(matching[0]?.getAttribute("data-message-error")).toBe("true");
  });

  it("does not duplicate a recovered inflight error when a late error event arrives", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      session_id: "runtime-1",
      running: false,
      messages: [],
      inflight: {
        user: "failed prompt",
        assistant: "partial answer",
        status: "error",
        error: "provider failure",
        turn_id: "failed-turn",
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("error", { text: "partial answer", error: "provider failure", turn_id: "failed-turn" }, "runtime-1"));

    const matching = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"))
      .filter((message) => message.textContent?.includes("partial answer"));
    expect(matching).toHaveLength(1);
    expect(matching[0]?.getAttribute("data-message-error")).toBe("true");
  });

  it("reconciles a message-complete error onto a durable row without a live assistant", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      session_id: "runtime-1",
      running: false,
      messages: [
        { id: "durable-user", role: "user", text: "durable prompt" },
        { id: "durable-assistant", role: "assistant", text: "partial answer" },
      ],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => {
      gateway.instance?.emit("message.complete", { status: "error", error: "provider failure", message_id: "durable-assistant" }, "runtime-1");
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    const target = assistants.filter((message) => message.dataset.messageId === "durable-assistant");
    expect(target).toHaveLength(1);
    expect(target[0]?.getAttribute("data-message-error")).toBe("true");
    expect(target[0]?.textContent).toContain("provider failure");
  });

  it("reconciles an error event onto a durable row without a live assistant", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      session_id: "runtime-1",
      running: false,
      messages: [
        { id: "durable-user", role: "user", text: "durable prompt" },
        { id: "durable-assistant", role: "assistant", text: "partial answer" },
      ],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("error", { error: "provider failure", message_id: "durable-assistant" }, "runtime-1"));

    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    const target = assistants.filter((message) => message.dataset.messageId === "durable-assistant");
    expect(target).toHaveLength(1);
    expect(target[0]?.getAttribute("data-message-error")).toBe("true");
    expect(target[0]?.textContent).toContain("provider failure");
  });

  it("prefers structured terminal errors over partial assistant text", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "error-turn" });
      gateway.instance?.emit("message.complete", { text: "partial answer", status: "error", error: "provider quota exceeded", turn_id: "error-turn" });
    });

    expect(host.querySelector("[data-slot='chat-error']")?.textContent).toContain("provider quota exceeded");
    expect(host.querySelector("[data-slot='chat-error']")?.textContent).not.toContain("partial answer");
    expect(host.querySelector("[data-message-error='true']")).toBeTruthy();
  });

  it("retains a terminal answer when message.start was missed", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.complete", { text: "final without start", turn_id: "missed-start" }));

    expect(host.textContent).toContain("final without start");
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(1);
  });

  it("keeps interim output as a separate transcript segment", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "interim-turn" });
      gateway.instance?.emit("message.interim", { text: "interim preview", turn_id: "interim-turn" });
      gateway.instance?.emit("message.delta", { text: "final answer", turn_id: "interim-turn" });
      gateway.instance?.emit("message.complete", { turn_id: "interim-turn" });
    });

    expect(host.querySelector("[data-message-interim='true']")?.textContent).toContain("interim preview");
    expect(host.textContent).toContain("final answer");
    expect(host.querySelector("[data-message-interim='true']")?.textContent).not.toContain("final answer");
  });

  it("blocks a seq-less retired Stop terminal after the replacement stream starts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn", message_id: "old-message" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn", message_id: "new-message" });
    });
    await act(async () => releaseInterrupt());
    await act(async () => gateway.instance?.emit("message.complete", { text: "stale old terminal" }));

    expect(host.textContent).toContain("new answer");
    expect(host.textContent).not.toContain("stale old terminal");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("does not retire a turn for a session-level error_surface", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "surface-turn", message_id: "surface-message" });
      gateway.instance?.emit("message.delta", { turn_id: "surface-turn", message_id: "surface-message", text: "live" });
      gateway.instance?.emit("error", { error: "session failure", error_surface: { layer: "gateway", code: "session" } });
    });
    expect(host.querySelector("[data-slot='turn-activity']")).not.toBeNull();
    expect(host.textContent).toContain("session failure");
  });

  it("rejects an old identified turn error while a new unbound stream is active", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-error-turn", message_id: "old-error-message" });
      gateway.instance?.emit("message.delta", { text: "old completed", turn_id: "old-error-turn", message_id: "old-error-message" });
      gateway.instance?.emit("message.complete", { turn_id: "old-error-turn", message_id: "old-error-message" });
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "current unbound" });
      gateway.instance?.emit("error", { turn_id: "old-error-turn", message_id: "old-error-message", error: "old error" });
    });

    expect(host.textContent).toContain("current unbound");
    expect(host.textContent).not.toContain("old error");
    expect(host.querySelector("button[aria-label='Stop']")).toBeTruthy();
  });

  it("rejects post-retirement unbound tool and interaction events", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "retired-control-turn", message_id: "retired-control-message" });
      gateway.instance?.emit("message.complete", { turn_id: "retired-control-turn", message_id: "retired-control-message", text: "done" });
      gateway.instance?.emit("tool.start", { name: "late tool" });
      gateway.instance?.emit("tool.generating", { name: "late generating" });
      gateway.instance?.emit("approval.request", { request_id: "late-approval", prompt: "late approval" });
      gateway.instance?.emit("clarify.request", { request_id: "late-clarify", question: "late clarify" });
    });
    expect(host.textContent).not.toContain("late tool");
    expect(host.textContent).not.toContain("late generating");
    expect(host.textContent).not.toContain("late approval");
    expect(host.textContent).not.toContain("late clarify");
  });

  it("splits an unbound stream when a message-id-only replacement starts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", {});
      gateway.instance?.emit("message.delta", { text: "old unbound" });
      gateway.instance?.emit("message.start", { message_id: "new-message-only" });
      gateway.instance?.emit("message.delta", { message_id: "new-message-only", text: "new answer" });
    });
    const assistants = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(assistants.filter((row) => row.textContent?.includes("old unbound"))).toHaveLength(1);
    expect(assistants.filter((row) => row.textContent?.includes("new answer"))).toHaveLength(1);
    expect(assistants).toHaveLength(2);
  });

  it("rejects post-retirement unbound delta before a new start boundary", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "retired-unbound-turn", message_id: "retired-unbound-message" });
      gateway.instance?.emit("message.delta", { text: "retired answer", turn_id: "retired-unbound-turn", message_id: "retired-unbound-message" });
      gateway.instance?.emit("message.complete", { turn_id: "retired-unbound-turn", message_id: "retired-unbound-message" });
      gateway.instance?.emit("message.delta", { text: "late unbound delta" });
    });
    expect(host.textContent).not.toContain("late unbound delta");
  });
  it("treats the string turn error_surface as turn-scoped", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "string-scoped error prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
      await Promise.resolve();
    });
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "string-scoped partial" });
      gateway.instance?.emit("error", { error_surface: "turn", error: "string-scoped failure" });
    });
    expect(host.querySelector("[data-message-role='assistant'][data-message-error='true']")).not.toBeNull();
    expect(host.textContent).toContain("string-scoped failure");
  });

  it("does not let an unbound session error terminate an active tagged turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "active-turn" });
      gateway.instance?.emit("message.delta", { text: "active answer", turn_id: "active-turn" });
      gateway.instance?.emit("error", { message: "session fault" });
    });

    expect(host.textContent).toContain("session fault");
    expect(host.textContent).toContain("active answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("ignores an unbound idle session.info while a current turn exists", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "active prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.start", { turn_id: "active-info-turn" });
      gateway.instance?.emit("message.delta", { text: "before stale idle", turn_id: "active-info-turn" });
      gateway.instance?.emit("session.info", { running: false });
      gateway.instance?.emit("message.delta", { text: " after stale idle", turn_id: "active-info-turn" });
    });

    expect(host.textContent).toContain("before stale idle after stale idle");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("does not treat an omitted session.info running field as idle", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "active-turn" });
      gateway.instance?.emit("message.delta", { text: "active answer", turn_id: "active-turn" });
      gateway.instance?.emit("session.info", { status: "thinking" });
    });

    expect(host.textContent).toContain("active answer");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
  });

  it("blocks retry when another turn becomes active after the failure", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "failed prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });
    const retry = host.querySelector<HTMLButtonElement>("button[aria-label='Retry send']")!;
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "other-turn" }));
    await act(async () => retry.click());

    expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "failed prompt")).toHaveLength(1);
    expect(host.querySelector("[data-slot='prompt-queue']")?.textContent).toContain("failed prompt");
  });

  it("ignores late events from a stopped same-session turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn" });
    });
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")?.click());
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn" });
      gateway.instance?.emit("message.delta", { text: " stale old answer", turn_id: "old-turn" });
      gateway.instance?.emit("message.complete", { text: "old complete", turn_id: "old-turn" });
    });

    expect(host.textContent).toContain("new answer");
    expect(host.textContent).not.toContain("stale old answer");
    expect(host.querySelector("[data-slot='turn-activity']")?.textContent).toContain("Thinking");
  });

  it("does not let an old Stop finally clear a newer Stop operation", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "old-turn" }));
    let releaseOld!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseOld = resolve; });
    const oldStop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { oldStop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });

    await act(async () => host.querySelector<HTMLButtonElement>("[data-testid='session-list'] button:last-child")?.click());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("message.start", { turn_id: "new-turn" }));
    let releaseNew!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseNew = resolve; });
    const newStop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { newStop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(newStop.disabled).toBe(true);

    await act(async () => {
      releaseOld();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(newStop.disabled).toBe(true);

    await act(async () => releaseNew());
  });

  it("does not let an unbound retired Stop terminal clear a replacement turn", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn" });
      gateway.instance?.emit("message.delta", { text: "old answer", turn_id: "old-turn" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { stop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });

    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn" });
      gateway.instance?.emit("message.complete", { text: "retired terminal" });
    });
    expect(host.textContent).toContain("new answer");
    expect(host.querySelector("[data-slot='turn-activity']")?.textContent).toContain("Thinking");
    await act(async () => releaseInterrupt());
  });

  it("retains a stopped turn's partial answer when a replacement turn completes first", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { stop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });

    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "new-turn" });
      gateway.instance?.emit("message.delta", { text: "new partial", turn_id: "new-turn" });
      gateway.instance?.emit("message.complete", { text: "new final", turn_id: "new-turn" });
    });
    expect(host.textContent).toContain("old partial");
    expect(host.textContent).toContain("new final");
    await act(async () => releaseInterrupt());
  });

  it("queues a replacement prompt until an in-flight Stop settles", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn" });
      gateway.instance?.emit("message.delta", { text: "old answer", turn_id: "old-turn" });
    });
    let releaseInterrupt!: () => void;
    gateway.instance!.interrupt = new Promise<void>((resolve) => { releaseInterrupt = resolve; });
    const stop = host.querySelector<HTMLButtonElement>("button[aria-label='Stop']")!;
    await act(async () => { stop.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => gateway.instance?.emit("message.complete", { text: "old final", turn_id: "old-turn" }));

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "replacement prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });
    expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "replacement prompt")).toHaveLength(0);

    await act(async () => releaseInterrupt());
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "replacement prompt")).toHaveLength(1);
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

  it("queues a prompt while a turn is running and submits it after completion", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.emit("message.start"));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "queued prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });

    expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "queued prompt")).toHaveLength(0);
    expect(host.querySelector("[data-slot='prompt-queue']")?.textContent).toContain("queued prompt");

    await act(async () => gateway.instance?.emit("message.complete"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "queued prompt")).toHaveLength(1);
  });

  it("resets queue drain ownership when the session effect is recreated by routing", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));
    await act(async () => gateway.instance?.emit("message.start"));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "first queued");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });

    let releaseSubmit!: () => void;
    let submitConsumed!: () => void;
    const consumed = new Promise<void>((resolve) => { submitConsumed = resolve; });
    gateway.instance!.submitBlock = new Promise<void>((resolve) => { releaseSubmit = resolve; });
    gateway.instance!.onSubmitBlockConsumed = submitConsumed;
    await act(async () => gateway.instance?.emit("message.complete"));
    await consumed;

    const reasoning = host.querySelector<HTMLSelectElement>("select[aria-label='Reasoning level']")!;
    await act(async () => {
      reasoning.value = "high";
      reasoning.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method }) => method === "session.create")).toHaveLength(2));
    await act(async () => { await Promise.resolve(); await new Promise((resolve) => setTimeout(resolve, 0)); });

    await act(async () => gateway.instance?.emit("message.start", { turn_id: "second-turn" }));
    await act(async () => {
      setter?.call(textarea, "second queued");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[type='submit']")?.click();
    });
    await act(async () => gateway.instance?.emit("message.complete", { turn_id: "second-turn" }));
    await vi.waitFor(() => expect(gateway.instance?.requests.filter(({ method, params }) => method === "prompt.submit" && params.text === "second queued")).toHaveLength(1));
    releaseSubmit();
  });

  it("opens the command palette with the platform shortcut and focuses the composer", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true })));
    expect(host.querySelector("[data-slot='command-palette']")).toBeTruthy();

    const focusComposer = host.querySelector<HTMLButtonElement>("button[aria-label='Focus composer']");
    expect(focusComposer).toBeTruthy();
    await act(async () => focusComposer?.click());
    expect(document.activeElement).toBe(host.querySelector("textarea"));
    expect(host.querySelector("[data-slot='command-palette']")).toBeNull();
  });

  it("does not open the command palette from an IME composing shortcut", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, isComposing: true, bubbles: true })));
    expect(host.querySelector("[data-slot='command-palette']")).toBeNull();
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

  it("automatically reconnects after an unexpected socket close", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const client = gateway.instance!;
    const initialConnectCalls = client.connectCalls;

    await act(async () => {
      client.stateHandler?.("closed");
      client.stateHandler?.("closed");
      await new Promise((resolve) => setTimeout(resolve, 300));
    });

    expect(client.connectCalls).toBe(initialConnectCalls + 1);
    expect(client.requests.filter(({ method }) => method === "session.activate")).toHaveLength(1);
  });

  it("cancels a pending reconnect when the chat page unmounts", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    const client = gateway.instance!;
    const initialConnectCalls = client.connectCalls;

    await act(async () => client.stateHandler?.("closed"));
    await act(async () => root.unmount());
    await new Promise((resolve) => setTimeout(resolve, 300));

    expect(client.connectCalls).toBe(initialConnectCalls);
  });

  it("re-attaches the same runtime session after reconnect", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(gateway.instance?.requests.map(({ method }) => method)).toContain("session.activate");
    expect(gateway.instance?.requests.find(({ method }) => method === "session.activate")?.params).toEqual({ session_id: "session-1", omit_messages: false, continue_on_disconnect: true, profile: "thai-profile" });
  });

  it("keeps a repeated inflight prompt separate without an active local anchor", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "repeat prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      gateway.instance?.emit("message.complete", { turn_id: "submitted-turn" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      messages: [],
      inflight: {
        user: "repeat prompt",
        assistant: "inflight answer",
        streaming: true,
        turn_id: "inflight-turn",
        message_id: "inflight-message",
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    const users = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='user']"));
    expect(users.filter((message) => message.textContent?.includes("repeat prompt"))).toHaveLength(2);
  });

  it("does not duplicate the user prompt when reconnect snapshot contains durable prompt and inflight continuation", async () => {
    gateway.MockGateway.initialResumeResponse = {
      session_id: "runtime-resume-dup",
      stored_session_id: "stored-resume-dup",
      running: true,
      status: "streaming",
      messages: [
        { id: 1, role: "user", text: "resumed prompt" },
        { id: 2, role: "assistant", text: "partial answer", streaming: true },
      ],
      inflight: {
        user: "resumed prompt",
        assistant: "partial answer continued",
        streaming: true,
        turn_id: "turn-dup",
        message_id: "msg-dup",
      },
    };
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=stored-resume-dup"] }, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(host.textContent).toContain("partial answer continued"));

    const userMessages = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='user']"));
    expect(userMessages.filter((el) => el.textContent?.includes("resumed prompt"))).toHaveLength(1);
  });

  it("hydrates an inflight reconnect snapshot and continues its partial answer", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "live-turn" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "live-turn" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      status: "streaming",
      inflight: { user: "inflight prompt", assistant: "durable partial", streaming: true, turn_id: "resume-turn", message_id: "resume-message" },
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("inflight prompt");
    expect(host.textContent).toContain("durable partial");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeTruthy();
    await act(async () => {
      gateway.instance?.emit("message.delta", { text: " continued", turn_id: "resume-turn", message_id: "resume-message" });
      gateway.instance?.emit("message.complete", { turn_id: "resume-turn", message_id: "resume-message" });
    });
    expect(host.textContent).toContain("durable partial continued");
  });

  it("preserves inflight correction bubbles and resumes the final segment", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      inflight: {
        user: "original prompt",
        assistant: "beforeafter",
        streaming: true,
        turn_id: "redirected-turn",
        message_id: "redirected-message",
        corrections: ["corrected prompt"],
        correction_offsets: [6],
      },
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("original prompt");
    expect(host.textContent).toContain("corrected prompt");
    expect(host.textContent).toContain("before");
    expect(host.textContent).toContain("after");
    expect(host.querySelectorAll("[data-message-role='assistant']")).toHaveLength(2);
    await act(async () => gateway.instance?.emit("message.delta", { text: " continued", turn_id: "redirected-turn", message_id: "redirected-message" }));
    expect(host.textContent).toContain("after continued");
  });

  it("retries an inflight corrected turn with its latest correction", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: false,
      inflight: {
        user: "original prompt",
        assistant: "partial answer",
        error: "provider failed",
        corrections: ["latest correction"],
        correction_offsets: [0],
        turn_id: "corrected-error-turn",
        message_id: "corrected-error-message",
      },
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(host.querySelector("button[aria-label='Retry send']")).toBeTruthy());
    await act(async () => host.querySelector<HTMLButtonElement>("button[aria-label='Retry send']")?.click());

    expect(gateway.instance?.requests.at(-1)).toEqual({
      method: "prompt.submit",
      params: { session_id: "session-1", text: "latest correction" },
    });
  });

  it("marks a status-only inflight failure as an error row", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: false,
      inflight: { user: "failed prompt", assistant: "partial answer", status: "error" },
      messages: [],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await vi.waitFor(() => expect(host.querySelector("[data-message-error='true']")).toBeTruthy());

    expect(host.textContent).toContain("Turn failed");
    expect(host.querySelector("button[aria-label='Retry send']")).toBeTruthy();
  });

  it("cleans the live turn when an inflight error snapshot omits running", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "live-turn" });
      gateway.instance?.emit("message.delta", { text: "live before error", turn_id: "live-turn" });
    });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      status: "error",
      messages: [],
      inflight: {
        user: "failed prompt",
        assistant: "failed partial",
        error: "provider failed",
        turn_id: "failed-turn",
        message_id: "failed-message",
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("provider failed");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.querySelector("button[aria-label='Stop']")).toBeNull();
    await act(async () => gateway.instance?.emit("message.delta", { text: "late mutation", turn_id: "live-turn" }));
    expect(host.textContent).not.toContain("late mutation");
  });

  it("clears stale live state when reconnect explicitly reports an idle session", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "stale-turn" });
      gateway.instance?.emit("message.delta", { text: "stale partial", turn_id: "stale-turn" });
      gateway.instance?.emit("tool.start", { tool_id: "stale-tool", name: "terminal", turn_id: "stale-turn" });
      gateway.instance?.emit("approval.request", { request_id: "stale-approval", command: "rm file", choices: ["once"], turn_id: "stale-turn" });
    });
    gateway.instance!.snapshot = { session_id: "session-1", running: false, status: "idle", messages: [], messages_omitted: false };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
    expect(host.querySelector("[aria-label='Approval required']")).toBeNull();
    expect(host.querySelector("[data-tool-id='stale-tool']")?.getAttribute("data-tool-state")).toBe("complete");
    expect(host.textContent).not.toContain("stale partial");
    await act(async () => {
      gateway.instance?.emit("message.start");
      gateway.instance?.emit("message.delta", { text: "fresh answer" });
    });
    expect(host.textContent).toContain("fresh answer");
    expect(host.textContent).not.toContain("stale partialfresh answer");
  });

  it("preserves live content for an empty idle snapshot without coverage", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "empty-idle-turn", message_id: "empty-idle-message" });
      gateway.instance?.emit("message.delta", { text: "empty idle partial", turn_id: "empty-idle-turn", message_id: "empty-idle-message" });
    });
    gateway.instance!.snapshot = { session_id: "session-1", running: false, messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.textContent).toContain("empty idle partial");
    expect(host.querySelector("[data-slot='turn-activity']")).toBeNull();
  });

  it("does not duplicate a turn-id-only inflight snapshot across reconnects", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      inflight: { user: "turn-only prompt", assistant: "partial", streaming: true, turn_id: "turn-only" },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      session_id: "session-1",
      running: true,
      inflight: { user: "turn-only prompt", assistant: "partial answer", streaming: true, turn_id: "turn-only" },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    const userRows = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='user']"));
    const assistantRows = Array.from(host.querySelectorAll<HTMLElement>("[data-message-role='assistant']"));
    expect(userRows.filter((row) => row.textContent?.includes("turn-only prompt"))).toHaveLength(1);
    expect(assistantRows.filter((row) => row.textContent?.includes("partial answer"))).toHaveLength(1);
  });

  it("does not resurrect a message-only turn after idle cleanup", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "retired-message" });
      gateway.instance?.emit("message.delta", { text: "stale partial", message_id: "retired-message" });
    });
    gateway.instance!.snapshot = { session_id: "session-1", running: false, messages: [] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "retired-message" });
      gateway.instance?.emit("message.delta", { text: "late replay", message_id: "retired-message" });
    });

    expect(host.textContent).not.toContain("late replay");
  });

  it("retires the previous identity after a tagged replacement start", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", turn_id: "old-turn", message_id: "old-message" });
      gateway.instance?.emit("message.start", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { text: "old resurrection", turn_id: "old-turn", message_id: "old-message" });
    });

    expect(host.textContent).toContain("new answer");
    expect(host.textContent).not.toContain("old resurrection");
  });

  it("retires the previous identity when an unbound stream becomes tagged", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await act(async () => {
      gateway.instance?.emit("message.start", { message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "old partial", message_id: "old-message" });
      gateway.instance?.emit("message.delta", { text: "new answer", turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { turn_id: "new-turn", message_id: "new-message" });
      gateway.instance?.emit("message.complete", { text: "old resurrection", message_id: "old-message" });
    });

    expect(host.textContent).toContain("new answer");
    expect(host.textContent).not.toContain("old resurrection");
  });

  it("does not duplicate a live assistant transcript when reconnect snapshot catches up", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(textarea, "recovered prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
    });
    await act(async () => {
      gateway.instance?.emit("message.start", undefined, "session-1", 1);
      gateway.instance?.emit("message.delta", { text: "recovered answer" }, "session-1", 2);
    });
    gateway.instance!.snapshot = { messages: [{ id: 9, role: "user", text: "recovered prompt" }, { id: 10, role: "assistant", text: "recovered answer" }] };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect((host.textContent?.match(/recovered answer/g) ?? []).length).toBe(1);
  });
  it("merges a truncated tail reconnect snapshot without reordering prior transcript", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => {
      expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true);
    });

    const textarea = host.querySelector<HTMLTextAreaElement>("textarea")!;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
    const submitTurn = async (prompt: string, answer: string) => {
      await act(async () => {
        setter?.call(textarea, prompt);
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter" }));
      });
      await act(async () => {
        gateway.instance?.emit("message.start");
        gateway.instance?.emit("message.delta", { text: answer });
        gateway.instance?.emit("message.complete");
      });
    };

    await submitTurn("first prompt", "first answer");
    await submitTurn("second prompt", "second answer");
    gateway.instance!.snapshot = {
      messages_omitted: true,
      messages: [
        { row_id: 2, id: 2, role: "user", text: "second prompt" },
        { row_id: 3, id: 3, role: "assistant", text: "second answer" },
      ],
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    const messages = Array.from(host.querySelectorAll<HTMLElement>("[data-slot='transcript-message']"));
    expect(messages.map((message) => message.dataset.messageRole)).toEqual([
      "user",
      "assistant",
      "user",
      "assistant",
    ]);
    expect(messages.map((message) => message.textContent)).toEqual([
      expect.stringContaining("first prompt"),
      expect.stringContaining("first answer"),
      expect.stringContaining("second prompt"),
      expect.stringContaining("second answer"),
    ]);
  });

  it("restores pending approval and clarification requests from a resume snapshot", async () => {
    await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ["/chat?resume=durable-1"] }, createElement(NativeChatPage))));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    gateway.instance!.snapshot = {
      running: true,
      status: "waiting",
      messages: [{ id: 7, role: "user", text: "previous prompt" }, { id: 8, role: "assistant", content: "previous answer" }],
      pending_approval: { request_id: "approval-resumed", command: "rm file", choices: ["once", "deny"], expires_at: 1 },
      pending_clarify: { request_id: "clarify-resumed", question: "Which?", choices: ["A"] },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain("rm file");
    expect(host.textContent).toContain("Which?");
    expect(host.textContent).toContain("Approval expired");
    expect(host.querySelector<HTMLButtonElement>("button[data-choice='once']")?.disabled).toBe(true);
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
      params: { close_on_disconnect: false, continue_on_disconnect: true, source: "dashboard", profile: "thai-profile" },
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
    const userArticle = host.querySelector<HTMLElement>("[data-message-role='user']");
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
    expect(gateway.instance?.requests[0]).toMatchObject({ method: "session.activate", params: { session_id: "durable-1", continue_on_disconnect: true, profile: "thai-profile" } });
    expect(host.textContent).toContain("previous prompt");
    expect(host.textContent).toContain("previous answer");
  });

  it("windows a large resumed transcript while preserving virtual spacer height", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    const largeSnapshot = { session_id: "session-1", messages: Array.from({ length: 120 }, (_, index) => ({ id: `row-${index}`, role: index % 2 === 0 ? "user" : "assistant", text: `message-${index}` })) };
    gateway.instance!.snapshot = largeSnapshot;
    await act(async () => {
      gateway.instance?.stateHandler?.("closed");
      gateway.instance?.stateHandler?.("open");
      await new Promise((resolve) => setTimeout(resolve, 20));
    });

    expect(gateway.instance?.requests.some(({ method }) => method === "session.activate")).toBe(true);
    const rows = host.querySelectorAll("[data-slot='transcript-row']");
    expect(rows.length).toBeLessThan(120);
    expect(host.querySelector("[data-slot='transcript-virtual-spacer']")?.getAttribute("data-total-height")).toBe("17280");
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
      close_on_disconnect: false, continue_on_disconnect: true, source: "dashboard", profile: "thai-profile",
    });
  });

  it("builds explicit model and reasoning payloads", () => {
    const choices = nativeChatModelChoices({ providers: [{ slug: "openai-codex", models: ["gpt-5.6-luna"] }] });
    expect(nativeChatSessionCreateParams(undefined, { model: choices[0], reasoning: "high" })).toEqual({
      close_on_disconnect: false, continue_on_disconnect: true, source: "dashboard", model: "gpt-5.6-luna", provider: "openai-codex", reasoning_effort: "high",
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

  it("restores a non-expired pending approval from reconnect snapshot, renders actionable Approve once, sends exact approval.respond parameters, and clears only after resolved 1", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method }) => method === "session.create")).toBe(true));

    const futureExpiry = Math.floor(Date.now() / 1000) + 3600;
    gateway.instance!.snapshot = {
      session_id: "session-1",
      pending_approval: {
        request_id: "snap-approval-1",
        command: "rm restored-file",
        choices: ["once", "deny"],
        expires_at: futureExpiry,
      },
    };
    await act(async () => gateway.instance?.stateHandler?.("closed"));
    await act(async () => gateway.instance?.stateHandler?.("open"));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    const approvalDialog = host.querySelector<HTMLElement>("[role='dialog']");
    expect(approvalDialog).not.toBeNull();
    expect(approvalDialog?.getAttribute("aria-label")).toBe("Approval required");
    expect(host.textContent).toContain("rm restored-file");
    expect(host.textContent).not.toContain("Approval expired");

    const approveButton = host.querySelector<HTMLButtonElement>("button[data-choice='once']");
    expect(approveButton).not.toBeNull();
    expect(approveButton?.disabled).toBe(false);
    expect(approveButton?.getAttribute("aria-label")).toBe("Approve once");

    // Verify that resolved: 0 does not clear the card and shows stale/error state
    gateway.instance!.approvalResponse = { resolved: 0 };
    await act(async () => approveButton?.click());
    expect(gateway.instance?.requests.at(-1)).toEqual({
      method: "approval.respond",
      params: {
        choice: "once",
        request_id: "snap-approval-1",
        session_id: "session-1",
        profile: "thai-profile",
      },
    });
    expect(host.querySelector("[role='dialog']")).not.toBeNull();
    expect(host.textContent).toContain("Approval was stale, expired, or already resolved");

    // Verify that card does not clear while in flight, and clears only after resolved: 1
    let releaseApproval!: () => void;
    let approvalStarted!: () => void;
    const approvalStartedPromise = new Promise<void>((resolve) => { approvalStarted = resolve; });
    gateway.instance!.approvalBlock = new Promise<void>((resolve) => { releaseApproval = resolve; });
    gateway.instance!.onApprovalBlockConsumed = approvalStarted;
    gateway.instance!.approvalResponse = { resolved: 1 };

    await act(async () => approveButton?.click());
    await approvalStartedPromise;

    expect(gateway.instance?.requests.at(-1)).toEqual({
      method: "approval.respond",
      params: {
        choice: "once",
        request_id: "snap-approval-1",
        session_id: "session-1",
        profile: "thai-profile",
      },
    });
    expect(host.querySelector("[role='dialog']")).not.toBeNull();

    await act(async () => {
      releaseApproval();
      await Promise.resolve();
    });

    expect(host.querySelector("[role='dialog']")).toBeNull();
    expect(host.textContent).not.toContain("rm restored-file");
  });

  it("clears and unmounts a visible pending approval card when switching profile scope without leaking it to the new profile", async () => {
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(NativeChatPage))));
    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method, params }) => method === "session.create" && params.profile === "thai-profile")).toBe(true));

    await act(async () => gateway.instance?.emit("approval.request", {
      request_id: "profile-leak-approval",
      command: "rm secret-profile-file",
      choices: ["once", "deny"],
    }));

    expect(host.querySelector("[role='dialog']")).not.toBeNull();
    expect(host.textContent).toContain("rm secret-profile-file");
    expect(host.querySelector("button[data-choice='once']")).not.toBeNull();

    await act(async () => {
      profileScopeState.current = "isolated-profile";
    });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });

    expect(host.querySelector("[role='dialog']")).toBeNull();
    expect(host.textContent).not.toContain("rm secret-profile-file");
    expect(host.querySelector("button[data-choice='once']")).toBeNull();

    await vi.waitFor(() => expect(gateway.instance?.requests.some(({ method, params }) => method === "session.create" && params.profile === "isolated-profile")).toBe(true));

    expect(host.querySelector("[role='dialog']")).toBeNull();
    expect(host.textContent).not.toContain("rm secret-profile-file");
  });
});
