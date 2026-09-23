import { describe, expect, it } from "vitest";

import {
  applyDashboardSubagentEvent,
  formatDashboardSubagentElapsed,
  listDashboardSubagents,
  type DashboardSubagentRoster,
} from "./dashboard-subagents";

const startA = {
  type: "subagent.start",
  payload: {
    subagent_id: "a",
    goal: "write",
    status: "running",
    started_at: 1700000000,
  },
} as const;

function runningA(): DashboardSubagentRoster {
  return applyDashboardSubagentEvent({}, startA);
}

describe("applyDashboardSubagentEvent", () => {
  it("creates a running row from a start event", () => {
    const next = applyDashboardSubagentEvent({}, startA);
    const rows = listDashboardSubagents(next);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      id: "a",
      status: "running",
      goal: "write",
    });
  });

  it("ignores events with no subagent_id (fail-open)", () => {
    const prev = {};
    const next = applyDashboardSubagentEvent(prev, {
      type: "subagent.start",
      payload: { goal: "x" },
    });
    expect(next).toEqual({});
    expect(next).toBe(prev);
  });

  it("maps unrecognized complete status to failed, not running", () => {
    const next = applyDashboardSubagentEvent(runningA(), {
      type: "subagent.complete",
      payload: { subagent_id: "a", status: "mystery" },
    });
    expect(listDashboardSubagents(next)[0]?.status).toBe("failed");
  });

  it("appends progress text and skips empty text", () => {
    let roster = runningA();
    roster = applyDashboardSubagentEvent(roster, {
      type: "subagent.progress",
      payload: { subagent_id: "a", text: "hello" },
    });
    expect(listDashboardSubagents(roster)[0]?.transcript).toEqual(["hello"]);

    roster = applyDashboardSubagentEvent(roster, {
      type: "subagent.progress",
      payload: { subagent_id: "a", text: "" },
    });
    expect(listDashboardSubagents(roster)[0]?.transcript).toEqual(["hello"]);

    roster = applyDashboardSubagentEvent(roster, {
      type: "subagent.progress",
      payload: { subagent_id: "a" },
    });
    expect(listDashboardSubagents(roster)[0]?.transcript).toEqual(["hello"]);
  });

  it("uses subagent_id as the label when goal is missing", () => {
    const next = applyDashboardSubagentEvent(
      {},
      {
        type: "subagent.spawn_requested",
        payload: { subagent_id: "child-1", status: "running" },
      },
    );
    expect(listDashboardSubagents(next)[0]?.goal).toBe("child-1");
  });

  it("keeps known complete statuses and ignores empty subagent_id", () => {
    let roster = runningA();
    roster = applyDashboardSubagentEvent(roster, {
      type: "subagent.complete",
      payload: { subagent_id: "a", status: "cancelled" },
    });
    expect(listDashboardSubagents(roster)[0]?.status).toBe("cancelled");

    const same = applyDashboardSubagentEvent(roster, {
      type: "subagent.complete",
      payload: { subagent_id: "", status: "completed" },
    });
    expect(same).toBe(roster);
  });
});

describe("formatDashboardSubagentElapsed", () => {
  it("shows an em dash when started_at is missing", () => {
    expect(formatDashboardSubagentElapsed(null)).toBe("—");
  });

  it("treats unix seconds as seconds when the value is below 1e12", () => {
    expect(formatDashboardSubagentElapsed(1_700_000_000 * 1000, 1_700_000_012 * 1000)).toBe(
      "12s",
    );
  });
});
