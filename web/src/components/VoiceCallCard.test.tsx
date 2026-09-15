// @vitest-environment jsdom
/**
 * Dashboard voice call card — the wiring the shared controller tests cannot
 * see: the sidecar-session turn counter that keys consult completion, the
 * mid-steer race (an interrupted turn completing while session.interrupt is
 * still in flight must not become the consult's answer), spoken stop
 * phrases, and foreign-session event isolation. The realtime socket client
 * is faked; the gateway client is mocked; the supervisor controller and the
 * stop-word matcher are the real shared implementations.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type CardCallbacks = {
  onFunctionCall: (call: {
    args: Record<string, unknown>;
    callId: string;
    name: string;
  }) => Promise<void> | void;
  onStatus?: (status: string, detail?: string) => void;
  onUserTranscript?: (text: string) => void;
};

type FakeClientInstance = {
  callbacks: CardCallbacks | null;
  closed: boolean;
  lastResponseHadAudio: boolean;
  outputs: [string, string][];
  spoken: string[];
};

const rtMocks = vi.hoisted(() => ({
  instances: [] as FakeClientInstance[],
}));

const gatewayMocks = vi.hoisted(() => {
  const handlers = new Map<string, (event: unknown) => void>();
  const requestHandlers: Array<(request: unknown) => boolean | void> = [];
  const state = {
    interrupt: undefined as undefined | (() => Promise<unknown>),
  };
  return {
    close: vi.fn(),
    connect: vi.fn(async () => undefined),
    handlers,
    on: vi.fn((event: string, handler: (event: unknown) => void) => {
      handlers.set(event, handler);
      return () => handlers.delete(event);
    }),
    onRequest: vi.fn((handler: (request: unknown) => boolean | void) => {
      requestHandlers.push(handler);
      return () => undefined;
    }),
    requestHandlers,
    request: vi.fn(async (method: string) => {
      if (method === "voice.realtime_token") {
        return {
          session_update: {},
          token: "eph-token",
          url: "wss://api.x.ai/v1/realtime?model=m",
        };
      }
      if (method === "session.create") {
        return { session_id: "sidecar-1" };
      }
      if (method === "session.interrupt" && state.interrupt) {
        return state.interrupt();
      }
      return {};
    }),
    state,
  };
});

vi.mock("@/lib/gatewayClient", () => ({
  GatewayClient: class {
    close = gatewayMocks.close;
    connect = gatewayMocks.connect;
    on = gatewayMocks.on;
    onRequest = gatewayMocks.onRequest;
    request = gatewayMocks.request;
  },
}));

// Only the socket client is faked — VoiceSupervisorController and the
// stop-word matcher stay the real shared implementations.
vi.mock("@hermes/shared", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@hermes/shared")>();
  class FakeRealtimeVoiceClient {
    callbacks: CardCallbacks | null = null;
    closed = false;
    lastResponseHadAudio = true;
    outputs: [string, string][] = [];
    spoken: string[] = [];

    constructor() {
      rtMocks.instances.push(this);
    }

    async connect(_grant: unknown, callbacks: CardCallbacks) {
      this.callbacks = callbacks;
    }

    close() {
      this.closed = true;
    }

    sendFunctionOutput(callId: string, output: string) {
      this.outputs.push([callId, output]);
    }

    speakAcknowledgment() {}
    speakVerbatim(text: string) {
      this.spoken.push(text);
    }
    setMuted() {}
  }
  return { ...actual, RealtimeVoiceClient: FakeRealtimeVoiceClient };
});

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  rtMocks.instances = [];
  gatewayMocks.handlers.clear();
  gatewayMocks.requestHandlers.length = 0;
  gatewayMocks.state.interrupt = undefined;
  vi.clearAllMocks();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

async function startCall(profile?: string): Promise<FakeClientInstance> {
  const { VoiceCallCard } = await import("./VoiceCallCard");
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<VoiceCallCard profile={profile} />));

  const button = container.querySelector("button");
  expect(button).not.toBeNull();
  await act(async () => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await vi.waitFor(() => {
    expect(rtMocks.instances).toHaveLength(1);
    expect(rtMocks.instances[0].callbacks).not.toBeNull();
  });
  return rtMocks.instances[0];
}

async function consult(client: FakeClientInstance, task: string, callId = "c1") {
  await act(async () => {
    void client.callbacks?.onFunctionCall({ args: { task }, callId, name: "consult_hermes" });
  });
  await vi.waitFor(() => {
    expect(gatewayMocks.request).toHaveBeenCalledWith(
      "prompt.submit",
      expect.objectContaining({ session_id: "sidecar-1", text: task }),
    );
  });
  // One more flush so the submit's pending-turn increment lands.
  await act(async () => undefined);
}

function completeTurn(text: string, sessionId = "sidecar-1") {
  gatewayMocks.handlers.get("message.complete")?.({
    payload: { text },
    session_id: sessionId,
    type: "message.complete",
  });
}

describe("VoiceCallCard", () => {
  it("mints the token, the session and the consult turns under the same scoped profile", async () => {
    const client = await startCall("coder");
    await consult(client, "check disk usage");

    for (const method of ["voice.realtime_token", "session.create", "prompt.submit"]) {
      expect(gatewayMocks.request).toHaveBeenCalledWith(method, expect.objectContaining({ profile: "coder" }));
    }
  });

  it("hands the finished turn's text to the voice model, ignoring foreign sessions", async () => {
    const client = await startCall();
    await consult(client, "check disk usage");

    // Another session's completion must not decrement or complete anything.
    await act(async () => completeTurn("other tenant", "someone-else"));
    expect(client.outputs).toEqual([]);

    await act(async () => completeTurn("Disk is 42% full."));
    expect(client.outputs).toContainEqual(["c1", "Disk is 42% full."]);
  });

  it("speaks a blocking prompt as a notice without answering the server request", async () => {
    const client = await startCall();
    await consult(client, "deploy the site");

    const respond = vi.fn();
    const outcomes = gatewayMocks.requestHandlers.map((handler) =>
      handler({
        fail: vi.fn(),
        id: "srq-1",
        method: "approval",
        params: { session_id: "sidecar-1" },
        respond,
      }),
    );

    expect(outcomes).toEqual([false]);
    expect(respond).not.toHaveBeenCalled();
    expect(client.spoken).toContain("Hermes needs your approval in the app.");
  });

  it("does not answer the consult with the interrupted turn's partial output", async () => {
    const client = await startCall();
    await consult(client, "original");

    let releaseInterrupt: () => void = () => undefined;
    gatewayMocks.state.interrupt = () =>
      new Promise((resolve) => {
        releaseInterrupt = () => resolve({});
      });

    // Steer while the consult's turn is running: the controller awaits
    // session.interrupt (held open here), then resubmits.
    await act(async () => {
      void client.callbacks?.onFunctionCall({
        args: { instruction: "also check logs" },
        callId: "s1",
        name: "steer_hermes",
      });
    });

    // The interrupted turn completes while the interrupt RPC is in flight —
    // the pending-turn counter hits zero, but this partial must not be
    // reported as the consult result.
    await act(async () => completeTurn("partial output"));
    expect(client.outputs).not.toContainEqual(["c1", "partial output"]);

    await act(async () => releaseInterrupt());
    await vi.waitFor(() => {
      expect(client.outputs).toContainEqual(["s1", "Steering applied — Hermes is adjusting course."]);
    });

    await act(async () => completeTurn("real answer"));
    expect(client.outputs).toContainEqual(["c1", "real answer"]);
  });

  it("ends the call on a spoken stop phrase, not only a bare 'stop'", async () => {
    const client = await startCall();
    await act(async () => client.callbacks?.onUserTranscript?.("Stop please."));
    expect(client.closed).toBe(true);
    expect(gatewayMocks.close).toHaveBeenCalled();
  });

  it("keeps listening on a substantive utterance that merely contains 'stop'", async () => {
    const client = await startCall();
    await act(async () => client.callbacks?.onUserTranscript?.("stop the docker container"));
    expect(client.closed).toBe(false);
  });
});
