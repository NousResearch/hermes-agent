// @vitest-environment node
import { describe, expect, it, vi } from "vitest";

import {
  promptSubmit,
  sessionCreate,
  sessionEventsSince,
  sessionHistory,
  sessionInterrupt,
  sessionResume,
  sessionSteer,
} from "@/lib/chat/gateway";

/**
 * Minimal `GatewayClient` surface used by the wrappers — we never touch the
 * real socket; the wrappers only call `request<...>(method, params)`.
 */
function fakeGw() {
  const request = vi.fn(async (_method: string, _params: unknown) => ({
    ok: true,
  }));
  return { request } as unknown as Parameters<typeof sessionResume>[0];
}

describe("chat gateway wrappers", () => {
  it("session.create accepts the full create-time param shape", async () => {
    const gw = fakeGw();
    const out = await sessionCreate(gw, {
      source: "dashboard",
      title: "New chat",
      model: "anthropic/claude-sonnet",
    });
    expect(out).toEqual({ ok: true });
    expect(gw.request).toHaveBeenCalledWith("session.create", {
      source: "dashboard",
      title: "New chat",
      model: "anthropic/claude-sonnet",
    });
  });

  it("session.resume forwards the contract params shape", async () => {
    const gw = fakeGw();
    const out = await sessionResume(gw, {
      session_id: "sess_123",
      cols: 80,
      lazy: true,
    });
    expect(out).toEqual({ ok: true });
    expect(gw.request).toHaveBeenCalledWith("session.resume", {
      session_id: "sess_123",
      cols: 80,
      lazy: true,
    });
  });

  it("session.history forwards only the session id", async () => {
    const gw = fakeGw();
    await sessionHistory(gw, { session_id: "sess_x" });
    expect(gw.request).toHaveBeenCalledWith("session.history", {
      session_id: "sess_x",
    });
  });

  it("session.interrupt supports an expected_hosted_task_id", async () => {
    const gw = fakeGw();
    await sessionInterrupt(gw, {
      session_id: "sess_x",
      expected_hosted_task_id: "task_42",
    });
    expect(gw.request).toHaveBeenCalledWith("session.interrupt", {
      session_id: "sess_x",
      expected_hosted_task_id: "task_42",
    });
  });

  it("prompt.submit forwards the exact contract shape — no images field", async () => {
    const gw = fakeGw();
    // NOTE: PromptSubmitParams does NOT have an `images` field. Image
    // attachment is a separate gateway call (image.attach / image.attach_bytes)
    // queued onto the next turn. This test pins the wrapper's surface so a
    // future refactor that "helpfully" adds images does not regress.
    await promptSubmit(gw, {
      session_id: "sess_x",
      text: "hello",
      surface: "dashboard",
    });
    expect(gw.request).toHaveBeenCalledWith("prompt.submit", {
      session_id: "sess_x",
      text: "hello",
      surface: "dashboard",
    });
  });

  it("session.events.since forwards last_seen seq", async () => {
    const gw = fakeGw();
    await sessionEventsSince(gw, { session_id: "sess_x", last_seen: 42 });
    expect(gw.request).toHaveBeenCalledWith("session.events.since", {
      session_id: "sess_x",
      last_seen: 42,
    });
  });

  it("session.steer forwards the SessionCorrectionParams shape", async () => {
    const gw = fakeGw();
    await sessionSteer(gw, {
      session_id: "sess_x",
      text: "actually, do it the other way",
    });
    expect(gw.request).toHaveBeenCalledWith("session.steer", {
      session_id: "sess_x",
      text: "actually, do it the other way",
    });
  });
});