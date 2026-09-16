import { describe, expect, it } from "vitest";

import {
  isRemoteApprovalExpired,
  normalizePendingApprovals,
  type PendingApprovalsResponse,
} from "./remote-approvals";

describe("remote approval normalization", () => {
  it("keeps only the selected profile, exact owner fields, and backend-authorized choices", () => {
    const response: PendingApprovalsResponse = {
      approvals: [
        {
          request_id: "req-work",
          command: "rm -rf /tmp/work",
          description: "delete work files",
          choices: ["once", "session", "always", "deny", "invented"],
          allow_permanent: false,
          profile: "work",
          session_id: "runtime-work",
          session_key: "stored-work",
          stored_session_id: "stored-work",
          source: "desktop",
          title: "Work session",
          status: "waiting_approval",
          created_at: 100,
          expires_at: 400,
          internal_secret: "must not be copied",
        },
        {
          request_id: "req-default",
          profile: "default",
          session_id: "runtime-default",
          session_key: "stored-default",
          stored_session_id: "stored-default",
        },
        {
          request_id: "missing-owner",
          profile: "work",
          command: "should be dropped",
        },
      ],
    };

    expect(normalizePendingApprovals(response, "work")).toEqual([{
      request_id: "req-work",
      command: "rm -rf /tmp/work",
      description: "delete work files",
      choices: ["once", "session", "deny"],
      allow_permanent: false,
      profile: "work",
      session_id: "runtime-work",
      session_key: "stored-work",
      stored_session_id: "stored-work",
      source: "desktop",
      title: "Work session",
      status: "waiting_approval",
      created_at: 100,
      expires_at: 400,
    }]);
  });

  it("does not treat a response for another profile as pending in the current scope", () => {
    expect(normalizePendingApprovals({
      approvals: [{
        request_id: "req-other",
        profile: "other",
        session_id: "runtime-other",
        session_key: "stored-other",
        stored_session_id: "stored-other",
        choices: ["once", "deny"],
      }],
    }, "work")).toEqual([]);
  });

  it("marks an approval expired from its server-provided absolute timestamp", () => {
    const [approval] = normalizePendingApprovals({
      approvals: [{
        request_id: "req-1",
        profile: "work",
        session_id: "runtime-work",
        session_key: "stored-work",
        stored_session_id: "stored-work",
        choices: ["once", "deny"],
        expires_at: 100,
      }],
    }, "work");

    expect(isRemoteApprovalExpired(approval, 100_000)).toBe(true);
    expect(isRemoteApprovalExpired(approval, 99_999)).toBe(false);
  });
});
