import { describe, expect, it } from "vitest";

import {
  buildOfficeAttentionItems,
  groupByText,
  visibleRows,
} from "./officeView";
import type { OfficeState } from "@/lib/api";

function officeFixture(overrides: Partial<OfficeState> = {}): OfficeState {
  return {
    schema_version: 1,
    generated_at: "2026-05-08T00:00:00Z",
    mode: "read_only",
    display_mode: "localhost",
    capabilities: {
      read_only: true,
      mutations_enabled: false,
      remote_mode: "unsupported",
    },
    data_sources: [],
    summary: {},
    rooms: [],
    agents: [],
    work_items: [],
    automations: [],
    topics: [],
    events: [],
    provenance: [],
    redactions: {
      policy_version: 1,
      redacted_field_count: 0,
      omitted_sections: [],
      warnings: [],
    },
    ...overrides,
  };
}

describe("OfficePage view helpers", () => {
  it("groups unknown work safely without reading sensitive body fields", () => {
    const grouped = groupByText([
      { id: "1", status: "blocked", title: "safe title", body: "raw body must not matter" },
      { id: "2", status: "", title: "missing status" },
    ], "status", "unknown");

    expect(Object.keys(grouped)).toEqual(["blocked", "unknown"]);
    expect(grouped.blocked).toHaveLength(1);
    expect(grouped.unknown).toHaveLength(1);
  });

  it("caps dense lists until show-more is requested", () => {
    const rows = Array.from({ length: 8 }, (_, index) => ({ id: index }));

    expect(visibleRows(rows, 6, false)).toHaveLength(6);
    expect(visibleRows(rows, 6, true)).toHaveLength(8);
  });

  it("builds attention rail from blocked work, failed automations, and unhealthy sources", () => {
    const attention = buildOfficeAttentionItems(
      officeFixture({
        data_sources: [
          {
            id: "topics",
            status: "missing",
            checked_at: "2026-05-08T00:00:00Z",
            warning_count: 0,
          },
          {
            id: "cron",
            status: "partial",
            checked_at: "2026-05-08T00:00:00Z",
            warning_count: 1,
          },
        ],
        work_items: [{ id: "task-1", title: "Blocked safe task", status: "blocked" }],
        automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", last_status: "error" }],
      }),
    );

    expect(attention.map((item) => item.id)).toEqual([
      "work:task-1",
      "automation:job-1",
      "source:cron",
    ]);
    expect(attention.map((item) => item.label).join(" ")).not.toContain("prompt");
  });
});
