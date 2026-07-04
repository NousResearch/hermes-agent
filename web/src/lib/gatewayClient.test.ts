import { describe, it, expect, vi, afterEach } from "vitest";
import { JsonRpcGatewayClient } from "@hermes/shared";

import { GatewayClient } from "./gatewayClient";

// The dashboard, the Electron desktop app, and the Ink TUI all drive the same
// tui_gateway JSON-RPC server, which historically stamped every turn
// platform="tui". GatewayClient now tags SESSION-ORIGINATING calls with
// source:"dashboard" so the backend attributes browser-dashboard turns to the
// dashboard client (→ blackbox turns.platform → tokens.ace source chart). These
// tests pin that contract by spying on the inherited request().

function captureForwarded() {
  // Spy on the base-class request so we see exactly what GatewayClient forwards,
  // without needing a live WebSocket.
  const spy = vi
    .spyOn(JsonRpcGatewayClient.prototype, "request")
    .mockResolvedValue({} as never);
  return spy;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GatewayClient source tagging", () => {
  it('stamps source:"dashboard" on session.create', async () => {
    const spy = captureForwarded();
    await new GatewayClient().request("session.create", { cols: 96 });
    expect(spy).toHaveBeenCalledWith(
      "session.create",
      { cols: 96, source: "dashboard" },
      undefined,
      undefined,
    );
  });

  it('stamps source:"dashboard" on session.resume', async () => {
    const spy = captureForwarded();
    await new GatewayClient().request("session.resume", { session_id: "s1" });
    expect(spy.mock.calls[0][1]).toMatchObject({ session_id: "s1", source: "dashboard" });
  });

  it("does NOT add source to non-session-originating methods", async () => {
    const spy = captureForwarded();
    await new GatewayClient().request("prompt.submit", { session_id: "s1", text: "hi" });
    expect(spy.mock.calls[0][1]).not.toHaveProperty("source");
  });

  it("never overrides an explicit caller-provided source (e.g. sidebar sidecar tool)", async () => {
    const spy = captureForwarded();
    await new GatewayClient().request("session.create", { close_on_disconnect: true, source: "tool" });
    expect(spy.mock.calls[0][1]).toMatchObject({ source: "tool" });
  });
});
