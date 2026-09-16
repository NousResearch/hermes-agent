import { describe, expect, it } from "vitest";
import {
  appendApprovalRequest,
  isNativeChatWorking,
  normalizeApprovalRequestPayload,
  reconnectActivateParams,
  removeApprovalRequest,
  shouldClearClarificationResponse,
} from "./NativeChatPage";

describe("Native Chat approval boundaries", () => {
  it("rejects malformed capability and choice data at the event boundary", () => {
    const request = normalizeApprovalRequestPayload({
      request_id: "rid-malformed",
      command: { secret: "fixture-secret" },
      description: ["raw-secret"],
      choices: { allow: "always" },
      allow_session: "yes",
      allow_permanent: 1,
    });
    expect(request).toEqual(expect.objectContaining({
      request_id: "rid-malformed",
      choices: ["deny"],
      allow_session: false,
      allow_permanent: false,
    }));
    expect(request?.command).toBeUndefined();
    expect(request?.description).toBeUndefined();
  });

  it("keeps valid choices, applies capability flags, and always retains deny", () => {
    const request = normalizeApprovalRequestPayload({
      request_id: "rid-valid",
      choices: ["once", "session", "always", "unknown"],
      allow_session: true,
      allow_permanent: false,
    });
    expect(request?.choices).toEqual(["once", "session", "deny"]);
    expect(request?.allow_session).toBe(true);
    expect(request?.allow_permanent).toBe(false);
  });

  it("queues distinct approvals and promotes by exact request ID", () => {
    const first = normalizeApprovalRequestPayload({ request_id: "rid-1", command: "first", choices: ["once", "deny"] })!;
    const second = normalizeApprovalRequestPayload({ request_id: "rid-2", command: "second", choices: ["once", "deny"] })!;
    const queue = appendApprovalRequest(appendApprovalRequest([], first), second);
    expect(queue.map((item) => item.request_id)).toEqual(["rid-1", "rid-2"]);
    expect(appendApprovalRequest(queue, first)).toBe(queue);
    const result = removeApprovalRequest(queue, "rid-1");
    expect(result.head?.request_id).toBe("rid-2");
    expect(result.queue.map((item) => item.request_id)).toEqual(["rid-2"]);
  });

  it("includes the selected profile in every reconnect activation payload", () => {
    expect(reconnectActivateParams("runtime-1", "work-profile")).toEqual({
      session_id: "runtime-1",
      omit_messages: false,
      continue_on_disconnect: true,
      profile: "work-profile",
    });
  });

  it("treats pending approval/clarification as an active cancellable turn", () => {
    expect(isNativeChatWorking({ streaming: false, runningTools: 0, turnStartedAt: null, hasPendingInteraction: true })).toBe(true);
    expect(isNativeChatWorking({ streaming: false, runningTools: 0, turnStartedAt: null, hasPendingInteraction: false })).toBe(false);
  });

  it("clears a batch clarification only after the final response", () => {
    expect(shouldClearClarificationResponse({ remaining: 1 }, "q-1")).toBe(false);
    expect(shouldClearClarificationResponse({ remaining: [] }, "q-1")).toBe(true);
    expect(shouldClearClarificationResponse({}, undefined)).toBe(true);
  });
});
