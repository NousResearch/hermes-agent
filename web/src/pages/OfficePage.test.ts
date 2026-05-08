import { describe, expect, it } from "vitest";

import {
  buildOfficeAttentionItems,
  buildOfficeMapFlows,
  buildOfficeMapNodes,
  buildOfficeSceneObjectView,
  buildOfficeSceneObjects,
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

  it("builds a browser-local office map from safe DTO counts", () => {
    const nodes = buildOfficeMapNodes(
      officeFixture({
        data_sources: [
          { id: "kanban", status: "ok", checked_at: "2026-05-08T00:00:00Z", item_count: 2 },
          { id: "topics", status: "missing", checked_at: "2026-05-08T00:00:00Z", item_count: 0 },
        ],
        agents: [{ id: "session-1", source_platform: "cli", status: "active", preview: "raw prompt must not matter" }],
        work_items: [{ id: "task-1", title: "Safe task", status: "blocked", body: "raw task body must not matter" }],
        automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", script: "raw script must not matter" }],
        topics: [],
      }),
    );

    expect(nodes.map((node) => node.id)).toEqual(["sessions", "work", "automation", "routing"]);
    expect(nodes.find((node) => node.id === "sessions")?.count).toBe(1);
    expect(nodes.find((node) => node.id === "work")?.count).toBe(1);
    expect(nodes.find((node) => node.id === "automation")?.count).toBe(1);
    expect(nodes.find((node) => node.id === "routing")?.health).toBe("missing");
    expect(nodes.map((node) => `${node.label} ${node.detail}`).join(" ")).not.toContain("raw");
  });


  it("builds safe office-map flow hints with degraded endpoint health", () => {
    const nodes = buildOfficeMapNodes(
      officeFixture({
        data_sources: [
          { id: "sessions", status: "ok", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
          { id: "kanban", status: "partial", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
          { id: "cron", status: "error", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
          { id: "topics", status: "missing", checked_at: "2026-05-08T00:00:00Z", item_count: 0 },
        ],
        agents: [{ id: "session-1", source_platform: "cli", status: "active", transcript: "raw transcript must not matter" }],
        work_items: [{ id: "task-1", title: "Safe task", status: "blocked", body: "raw task body must not matter" }],
        automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", script: "raw script must not matter" }],
      }),
    );
    const flows = buildOfficeMapFlows(nodes);

    expect(flows.map((flow) => `${flow.from}->${flow.to}`)).toEqual([
      "sessions->work",
      "work->automation",
      "automation->routing",
    ]);
    expect(flows.map((flow) => flow.health)).toEqual(["partial", "error", "error"]);
    expect(flows.map((flow) => flow.label).join(" ")).not.toContain("raw");
    expect(nodes.map((node) => node.zone)).toEqual(["entry", "workbench", "machine", "routing"]);
    expect(nodes.every((node) => node.x >= 12 && node.x <= 88 && node.y >= 16 && node.y <= 84)).toBe(true);
  });

  it("builds safe 2D office scene objects with caps and no raw field projection", () => {
    const state = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-08T00:00:00Z", item_count: 8 },
        { id: "kanban", status: "partial", checked_at: "2026-05-08T00:00:00Z", item_count: 7 },
        { id: "cron", status: "error", checked_at: "2026-05-08T00:00:00Z", item_count: 2 },
        { id: "topics", status: "missing", checked_at: "2026-05-08T00:00:00Z", item_count: 0 },
      ],
      agents: Array.from({ length: 8 }, (_, index) => ({ id: `session-${index}`, source_platform: "cli", status: "active", transcript: "raw transcript must not matter" })),
      work_items: Array.from({ length: 7 }, (_, index) => ({ id: `task-${index}`, title: "Safe task", status: index === 0 ? "blocked" : "open", body: "raw task body must not matter" })),
      automations: [
        { id: "job-1", name: "Cron job job-1", state: "scheduled", last_status: "ok", script: "raw script must not matter" },
        { id: "job-2", name: "Cron job job-2", state: "error", last_status: "error", secret: "raw secret must not matter" },
      ],
      topics: [],
      provenance: [],
    });
    const nodes = buildOfficeMapNodes(state);
    const objects = buildOfficeSceneObjects(state, nodes);

    expect(objects.map((object) => object.kind)).toContain("avatar");
    expect(objects.map((object) => object.kind)).toContain("desk");
    expect(objects.map((object) => object.kind)).toContain("machine");
    expect(objects.map((object) => object.kind)).toContain("mail");
    expect(objects.filter((object) => object.roomId === "sessions" && object.kind === "avatar")).toHaveLength(6);
    expect(objects.find((object) => object.id === "sessions-overflow")?.label).toBe("+2 세션");
    expect(objects.find((object) => object.id === "work-overflow")?.label).toBe("+1 작업");
    expect(objects.find((object) => object.roomId === "routing")?.label).toBe("미연결 보관함");
    expect(objects.every((object) => object.x >= 10 && object.x <= 90 && object.y >= 12 && object.y <= 88)).toBe(true);
    expect(objects.map((object) => `${object.label} ${object.detail}`).join(" ")).not.toMatch(/raw|transcript|body|script|secret/i);
  });

  it("builds non-interactive accessible marker presentation without raw details", () => {
    const marker = buildOfficeSceneObjectView({
      id: "work-desk-1",
      roomId: "work",
      kind: "desk",
      label: "work desk 1",
      detail: "workbench safe marker",
      health: "partial",
      x: 63,
      y: 21,
    });

    expect(marker.glyph).toBe("▤");
    expect(marker.title).toBe("work desk 1 · workbench safe marker");
    expect(marker.ariaHidden).toBe(true);
    expect(marker.interactive).toBe(false);
    expect(marker.toneClass).toContain("yellow");
    expect(marker.title).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

});
