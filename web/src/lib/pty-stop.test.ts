import { describe, expect, it, vi } from "vitest";
import { createPtyStop } from "./pty-stop";

describe("PTY stop protocol", () => {
  it("rejects unsolicited state before confirmation and stale sockets, and waits for the stop reply", () => {
    const send = vi.fn();
    let current = true;
    const stop = createPtyStop(() => {});
    stop.open({ readyState: 1, send }, () => current);
    const nonce = JSON.parse(send.mock.calls[0][0].slice(7, -1)).nonce;
    const bound = { sid: "s", epoch: "e", generation: 2 };
    stop.receive(JSON.stringify({ action: "state", bound }));
    expect(stop.enabled()).toBe(false);
    stop.receive(JSON.stringify({ action: "state", nonce, bound }));
    stop.stop();
    stop.receive(JSON.stringify({ action: "state", bound }));
    expect(stop.enabled()).toBe(false);
    stop.receive(JSON.stringify({ action: "state", nonce, bound }));
    expect(stop.enabled()).toBe(true);
    current = false;
    expect(stop.enabled()).toBe(false);
    stop.stop();
    expect(send).toHaveBeenCalledTimes(2);
    stop.close();
    stop.receive(JSON.stringify({ action: "state", nonce, bound }));
    expect(stop.enabled()).toBe(false);
  });

  it("reports send failures and rejects malformed bindings", () => {
    const send = vi.fn();
    const stop = createPtyStop(() => {});
    stop.open({ readyState: 1, send });
    const nonce = JSON.parse(send.mock.calls[0][0].slice(7, -1)).nonce;
    stop.receive(JSON.stringify({ action: "state", nonce, bound: { sid: "s", epoch: "e", generation: 1.5 } }));
    expect(stop.enabled()).toBe(false);
    stop.receive(JSON.stringify({ action: "state", nonce, bound: { sid: "s", epoch: "e", generation: 1 } }));
    send.mockImplementation(() => { throw new Error("closed"); });
    stop.stop();
    expect(stop.error()).toContain("Could not send");
  });

  it("queries and only stops a confirmed current binding", () => {
    const send = vi.fn();
    const changed = vi.fn();
    const socket = { readyState: 1, send };
    const stop = createPtyStop(changed);
    stop.open(socket);
    const query = JSON.parse(send.mock.calls[0][0].slice(7, -1));
    expect(query.action).toBe("query");
    expect(query.nonce).toMatch(/^[0-9a-f]{32}$/);
    stop.receive(JSON.stringify({ action: "state", nonce: "wrong", bound: { sid: "s", epoch: "e", generation: 1 } }));
    expect(stop.enabled()).toBe(false);
    stop.receive(JSON.stringify({ action: "state", nonce: query.nonce, bound: { sid: "s", epoch: "e", generation: 1 } }));
    expect(stop.enabled()).toBe(true);
    stop.stop();
    expect(JSON.parse(send.mock.calls[1][0].slice(7, -1))).toEqual({ action: "stop", sid: "s", epoch: "e", generation: 1, nonce: query.nonce });
    expect(stop.enabled()).toBe(false);
    stop.receive(JSON.stringify({ action: "state", nonce: query.nonce, bound: null }));
    expect(stop.enabled()).toBe(false);
  });
});
