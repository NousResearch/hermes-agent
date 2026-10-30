// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => {
  const handlers = new Map<string, Set<(event: unknown) => void>>();
  const requests: { method: string; params: Record<string, unknown> }[] = [];
  return { handlers, requests };
});

vi.mock("@/lib/gatewayClient", () => {
  class FakeGatewayClient {
    on(type: string, handler: (event: unknown) => void) {
      const set = harness.handlers.get(type) ?? new Set();
      set.add(handler);
      harness.handlers.set(type, set);
      return () => set.delete(handler);
    }
    onState(handler: (state: string) => void) {
      handler("open");
      return () => undefined;
    }
    onRequest() {
      return () => undefined;
    }
    async connect() {}
    async request(method: string, params: Record<string, unknown> = {}) {
      harness.requests.push({ method, params });
      if (method === "session.create" || method === "session.resume") {
        return { session_id: "s1", messages: [] };
      }
      return {};
    }
    close() {}
  }
  return { GatewayClient: FakeGatewayClient };
});

import { NativeChat } from "./NativeChat";

function emit(type: string, payload: unknown) {
  for (const handler of harness.handlers.get(type) ?? []) {
    handler({ type, payload, session_id: "s1", seq: 1 });
  }
}

describe("NativeChat", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    harness.handlers.clear();
    harness.requests.length = 0;
    localStorage.clear();
    Element.prototype.scrollIntoView = () => undefined;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("hydrates a live session and renders an adviser event as a card", async () => {
    await act(async () => {
      root.render(<NativeChat onExit={() => undefined} />);
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(harness.requests.some((call) => call.method === "session.create")).toBe(true);

    await act(async () => {
      emit("message.adviser", {
        id: "ad-1",
        text: "[adviser] check the migration before deploy",
        severity: "blocker",
        label: "adviser · blocker",
        display_kind: "adviser",
      });
    });

    expect(container.textContent).toContain("check the migration before deploy");
    expect(container.textContent).toContain("adviser · blocker");
    expect(container.textContent).toContain("Terminal");
  });
});
