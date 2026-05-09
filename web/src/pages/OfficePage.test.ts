import { describe, expect, it } from "vitest";

import {
  buildOfficeAttentionItems,
  buildOfficeCharacterActivity,
  buildOfficeCharacterInspector,
  buildOfficeCharacterRoutes,
  buildOfficeCharacterSceneObjects,
  buildOfficeCharacterView,
  buildOfficeCharacters,
  buildOfficeMapDensityPlan,
  buildOfficeMapJumpTargets,
  buildOfficeMapPolishPlan,
  buildOfficeResponsiveReadabilityPlan,
  buildOfficeMapFlows,
  buildOfficeMapNodes,
  buildOfficeSceneMotionTrack,
  buildOfficeSceneObjectView,
  buildOfficeSceneObjects,
  buildOfficeStateDelta,
  buildOfficeAutomationTimingSummary,
  buildOfficeEmptySourceCopyPlan,
  buildOfficeEmptyStateHints,
  buildOfficeSourceHealthSummary,
  buildOfficeUsabilitySummary,
  mergeOfficeRecentChanges,
  resolveOfficeLiveTrackingInterval,
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

  it("builds safe CSS motion tracks so the 2D scene feels alive without a renderer", () => {
    const avatarMotion = buildOfficeSceneMotionTrack({
      id: "sessions-avatar-1",
      roomId: "sessions",
      kind: "avatar",
      label: "세션 표시 1",
      detail: "entry 안전 표시",
      health: "ok",
      x: 17,
      y: 22,
    });
    const machineMotion = buildOfficeSceneMotionTrack({
      id: "automation-machine-1",
      roomId: "automation",
      kind: "machine",
      label: "자동화 기계 1",
      detail: "machine 안전 표시",
      health: "partial",
      x: 17,
      y: 58,
    });

    expect(avatarMotion).toMatchObject({
      className: "office-scene-marker-motion office-scene-marker-walk",
      ariaLabel: "세션 표시 1 이동 표시 · 안전 DTO 기반",
    });
    expect(avatarMotion.style).toMatchObject({
      "--office-motion-x": "3px",
      "--office-motion-y": "-2px",
      "--office-motion-duration": "4.8s",
      "--office-motion-delay": "-0.3s",
    });
    expect(machineMotion.className).toBe("office-scene-marker-motion office-scene-marker-blink");
    expect(machineMotion.ariaLabel).toBe("자동화 기계 1 상태등 표시 · 안전 DTO 기반");
    expect(`${avatarMotion.ariaLabel} ${Object.values(avatarMotion.style).join(" ")} ${machineMotion.ariaLabel}`).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

  it("builds safe RPG role characters for models and office work without raw projection", () => {
    const state = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-09T00:00:00Z", item_count: 2 },
        { id: "kanban", status: "partial", checked_at: "2026-05-09T00:00:00Z", item_count: 4 },
        { id: "cron", status: "ok", checked_at: "2026-05-09T00:00:00Z", item_count: 1 },
        { id: "topics", status: "missing", checked_at: "2026-05-09T00:00:00Z", item_count: 0 },
      ],
      agents: [
        { id: "agent-1", model: "safe-model", status: "active", prompt: "raw prompt must not matter" },
        { id: "agent-2", provider: "safe-provider", status: "idle", transcript: "raw transcript must not matter" },
      ],
      work_items: Array.from({ length: 4 }, (_, index) => ({ id: `task-${index}`, title: "Safe task", status: index === 0 ? "blocked" : "open", body: "raw task body must not matter" })),
      automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", next_run_at: "2026-05-09T01:00:00Z", script: "raw script must not matter" }],
      topics: [],
      provenance: [],
    });
    const characters = buildOfficeCharacters(state, buildOfficeMapNodes(state));

    expect(characters.map((character) => character.role)).toEqual(expect.arrayContaining(["model", "worker", "automation_keeper", "router", "sentinel", "alert"]));
    expect(characters.find((character) => character.role === "model")?.label).toBe("모델 캐릭터 1");
    expect(characters.find((character) => character.role === "worker")?.label).toBe("작업자 1");
    expect(characters.find((character) => character.role === "automation_keeper")?.label).toBe("자동화 관리인 1");
    expect(characters.find((character) => character.role === "router")?.label).toBe("전달자 1");
    expect(characters.find((character) => character.role === "sentinel")?.label).toBe("감시자 1");
    expect(characters.find((character) => character.role === "alert")?.status).toBe("blocked");
    expect(characters.every((character) => character.roomId && character.status && character.redactionNote && character.x >= 10 && character.x <= 90 && character.y >= 12 && character.y <= 88)).toBe(true);
    expect(characters.map((character) => `${character.label} ${character.detail} ${character.redactionNote}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret|safe-model|safe-provider/i);
  });

  it("builds original CSS character views for each RPG role without raw labels", () => {
    const roles = ["model", "operator", "worker", "reviewer", "automation_keeper", "router", "sentinel", "alert"] as const;
    const views = roles.map((role, index) =>
      buildOfficeCharacterView({
        id: `${role}-1`,
        role,
        roomId: role === "automation_keeper" ? "automation" : role === "router" || role === "sentinel" ? "routing" : role === "model" || role === "operator" ? "sessions" : "work",
        label: `raw ${role} model prompt ${index}`,
        status: role === "alert" ? "blocked" : role === "reviewer" ? "reviewing" : role === "router" ? "routing" : role === "automation_keeper" ? "scheduled" : "active",
        detail: "raw transcript body script secret must not matter",
        redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
        x: 20,
        y: 20,
      }),
    );

    expect(views.map((view) => view.nameplate)).toEqual(["모델", "조작", "작업", "검토", "자동화", "전달", "감시", "경보"]);
    expect(views.every((view) => view.bodyClassName.startsWith("office-character office-character--"))).toBe(true);
    expect(views.every((view) => view.accessoryClassName.startsWith("office-character__accessory office-character__accessory--"))).toBe(true);
    expect(views.map((view) => view.statusLabel)).toEqual(["활성", "활성", "활성", "검토", "예약", "전달", "활성", "막힘"]);
    expect(views.map((view) => `${view.glyph} ${view.nameplate} ${view.statusLabel} ${view.safeTitle} ${view.bodyClassName} ${view.accessoryClassName}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret|model prompt/i);
  });

  it("builds safe action loops for RPG characters from status and room delta only", () => {
    const emptyDelta = buildOfficeStateDelta(null, officeFixture());
    const changedDelta = {
      ...emptyDelta,
      nodeBadges: {
        ...emptyDelta.nodeBadges,
        automation: [{ label: "일정 변경", tone: "warning" as const }],
        routing: [{ label: "상태 변경", tone: "warning" as const }],
      },
    };
    const base = {
      id: "character-1",
      roomId: "sessions" as const,
      label: "raw prompt transcript body script secret must not matter",
      detail: "raw model provider task body must not matter",
      redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
      x: 20,
      y: 20,
    };

    const activities = [
      buildOfficeCharacterActivity({ ...base, role: "model", status: "active" }, emptyDelta),
      buildOfficeCharacterActivity({ ...base, role: "worker", roomId: "work", status: "working" }, emptyDelta),
      buildOfficeCharacterActivity({ ...base, role: "reviewer", roomId: "work", status: "reviewing" }, emptyDelta),
      buildOfficeCharacterActivity({ ...base, role: "router", roomId: "routing", status: "routing" }, changedDelta),
      buildOfficeCharacterActivity({ ...base, role: "automation_keeper", roomId: "automation", status: "scheduled" }, emptyDelta),
      buildOfficeCharacterActivity({ ...base, role: "automation_keeper", roomId: "automation", status: "scheduled" }, changedDelta),
      buildOfficeCharacterActivity({ ...base, role: "alert", roomId: "work", status: "blocked" }, emptyDelta),
      buildOfficeCharacterActivity({ ...base, role: "sentinel", roomId: "routing", status: "warning" }, emptyDelta),
      buildOfficeCharacterActivity({ ...base, role: "sentinel", roomId: "routing", status: "unknown" }, emptyDelta),
      buildOfficeCharacterActivity({ ...base, role: "operator", status: "idle" }, emptyDelta),
    ];

    expect(activities.map((activity) => activity.id)).toEqual(["thinking", "working", "reviewing", "routing", "scheduled", "soon", "blocked", "warning", "unknown", "idle"]);
    expect(activities.map((activity) => activity.label)).toEqual(["생각 중", "작업 중", "검토 중", "전달 중", "예약 대기", "곧 실행", "막힘", "확인 필요", "확인 불가", "대기"]);
    expect(activities.map((activity) => activity.tone)).toEqual(["normal", "success", "success", "normal", "muted", "warning", "danger", "warning", "muted", "muted"]);
    expect(activities.map((activity) => activity.motion)).toEqual(["idle", "pulse", "idle", "walk", "idle", "blink", "blink", "blink", "none", "none"]);
    expect(activities.map((activity) => `${activity.id} ${activity.label} ${activity.reducedMotionLabel}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret|model|provider/i);
  });

  it("builds safe room-to-room RPG routes from changed flows only", () => {
    const delta = {
      ...buildOfficeStateDelta(null, officeFixture()),
      changedFlows: [
        { from: "sessions" as const, to: "work" as const, label: "세션에서 작업으로", tone: "positive" as const },
        { from: "work" as const, to: "automation" as const, label: "raw prompt transcript body script secret must not matter", tone: "warning" as const },
        { from: "automation" as const, to: "routing" as const, label: "자동화에서 라우팅으로", tone: "negative" as const },
      ],
    };

    const routes = buildOfficeCharacterRoutes(delta);

    expect(routes.map((route) => route.id)).toEqual([
      "route:sessions->work",
      "route:work->automation",
      "route:automation->routing",
    ]);
    expect(routes.map((route) => route.label)).toEqual(["흐름 변경", "흐름 변경", "흐름 변경"]);
    expect(routes.map((route) => route.detail)).toEqual([
      "세션에서 작업으로 · 방금 변경",
      "작업에서 자동화로 · 방금 변경",
      "자동화에서 라우팅으로 · 방금 변경",
    ]);
    expect(routes.map((route) => route.motion)).toEqual(["route", "alert", "alert"]);
    expect(routes.map((route) => route.tone)).toEqual(["normal", "warning", "danger"]);
    expect(routes.map((route) => `${route.label} ${route.detail} ${route.reducedMotionLabel}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

  it("builds safe character inspector fields for keyboard-accessible RPG character inspection", () => {
    const delta = {
      ...buildOfficeStateDelta(null, officeFixture()),
      nodeBadges: {
        sessions: [{ label: "+2", tone: "positive" as const }],
        work: [],
        automation: [],
        routing: [],
      },
      changedFlows: [{ from: "sessions" as const, to: "work" as const, label: "raw prompt transcript body script secret must not matter", tone: "warning" as const }],
    };
    const character = {
      id: "model-1",
      role: "model" as const,
      roomId: "sessions" as const,
      label: "raw model provider prompt must not matter",
      status: "active" as const,
      detail: "raw transcript body script secret must not matter",
      redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
      x: 20,
      y: 20,
    };

    const inspector = buildOfficeCharacterInspector(character, delta);

    expect(inspector.kind).toBe("RPG 캐릭터");
    expect(inspector.title).toBe("세션 · 모델 캐릭터");
    expect(inspector.ariaLabel).toBe("모델 캐릭터 살펴보기, 방 세션, 상태 활성, 액션 생각 중");
    expect(inspector.fields).toEqual([
      ["캐릭터", "모델 캐릭터"],
      ["역할", "모델"],
      ["방", "세션"],
      ["상태", "활성"],
      ["액션", "생각 중"],
      ["최근 안전 변화", "+2 · 세션에서 작업으로 · 방금 변경"],
      ["가림", "안전 DTO 역할/상태/개수/흐름만 반영 · 원문 제외"],
    ]);
    expect(`${inspector.title} ${inspector.ariaLabel} ${inspector.fields.flat().join(" ")}`).not.toMatch(/raw|prompt|transcript|body|script|secret|model provider/i);
  });

  it("builds Stage 10-F usability summary for dense, missing-source, reduced-motion, and responsive states", () => {
    const state = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-09T00:00:00Z", item_count: 8 },
        { id: "kanban", status: "partial", checked_at: "2026-05-09T00:00:00Z", item_count: 8, warning_count: 1, error_summary: "raw stack secret must not matter" },
        { id: "cron", status: "ok", checked_at: "2026-05-09T00:00:00Z", item_count: 5 },
      ],
      agents: Array.from({ length: 8 }, (_, index) => ({ id: `agent-${index}`, status: "active", prompt: "raw prompt must not matter" })),
      work_items: Array.from({ length: 8 }, (_, index) => ({ id: `task-${index}`, title: "Safe task", status: index === 0 ? "blocked" : "open", body: "raw body must not matter" })),
      automations: Array.from({ length: 5 }, (_, index) => ({ id: `job-${index}`, state: "scheduled", script: "raw script must not matter" })),
      topics: [],
      provenance: [],
    });
    const nodes = buildOfficeMapNodes(state);
    const characters = buildOfficeCharacters(state, nodes);

    const summary = buildOfficeUsabilitySummary(state, characters, { reducedMotion: true, viewportWidth: 430 });

    expect(summary.items.map((item) => item.id)).toEqual(["density", "source-fallback", "motion", "responsive", "korean-copy"]);
    expect(summary.items.map((item) => item.label)).toEqual(["밀도 점검", "소스 공백", "동작 모드", "좁은 화면", "한국어 우선"]);
    expect(summary.items.find((item) => item.id === "density")?.detail).toContain("합산 캐릭터");
    expect(summary.items.find((item) => item.id === "source-fallback")?.detail).toContain("미연결/부분 연결");
    expect(summary.items.find((item) => item.id === "motion")?.detail).toContain("정지");
    expect(summary.items.find((item) => item.id === "responsive")?.detail).toContain("세로");
    expect(summary.items.every((item) => item.detail.length > 0 && item.tone)).toBe(true);
    expect(summary.items.map((item) => `${item.label} ${item.detail}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret|stack/i);
  });

  it("builds Stage 10-G safe readability density plans without exposing raw character details", () => {
    const characters = Array.from({ length: 14 }, (_, index) => ({
      id: `character-${index}`,
      role: "model" as const,
      roomId: "sessions" as const,
      label: index === 13 ? "raw prompt transcript secret must not matter" : `모델 캐릭터 ${index + 1}`,
      status: "active" as const,
      detail: "raw task body script must not matter",
      redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
      x: 20,
      y: 20,
    }));

    const summary = buildOfficeMapDensityPlan("summary", characters);
    const standard = buildOfficeMapDensityPlan("standard", characters);
    const detail = buildOfficeMapDensityPlan("detail", characters);

    expect([summary.label, standard.label, detail.label]).toEqual(["요약", "표준", "상세"]);
    expect(summary.visibleCharacters).toHaveLength(6);
    expect(standard.visibleCharacters).toHaveLength(12);
    expect(detail.visibleCharacters).toHaveLength(14);
    expect(summary.hiddenCharacterCount).toBe(8);
    expect(standard.hiddenCharacterCount).toBe(2);
    expect(detail.hiddenCharacterCount).toBe(0);
    expect(summary.showUsabilityRail).toBe(true);
    expect(summary.showRecentRail).toBe(false);
    expect(standard.showRecentRail).toBe(true);
    expect(detail.showRecentRail).toBe(true);
    expect(`${summary.label} ${summary.detail} ${summary.visibleCharacters.map((character) => character.label).join(" ")}`).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

  it("builds Stage 11-B CSS/SVG polish plans for crowded labels and lower rails", () => {
    const characters = Array.from({ length: 14 }, (_, index) => ({
      id: `character-${index}`,
      role: "automation_keeper" as const,
      roomId: index < 8 ? ("automation" as const) : ("sessions" as const),
      label: index === 13 ? "raw prompt transcript secret must not matter" : `자동화 관리인 ${index + 1}`,
      status: index % 3 === 0 ? ("scheduled" as const) : ("active" as const),
      detail: "raw task body script must not matter",
      redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
      x: 20,
      y: 20,
    }));

    const standardPlan = buildOfficeMapDensityPlan("standard", characters);
    const summaryPlan = buildOfficeMapDensityPlan("summary", characters);
    const standardPolish = buildOfficeMapPolishPlan(standardPlan);
    const summaryPolish = buildOfficeMapPolishPlan(summaryPlan);

    expect(standardPolish).toMatchObject({
      stageLabel: "Stage 11-B 정돈",
      characterLabelMode: "compact",
      lowerRailMode: "detached",
      mapClassName: "office-map--polished office-map--labels-compact office-map--rail-detached",
      legendClassName: "office-map-legend office-map-legend--detached",
    });
    expect(standardPolish.notes).toEqual(expect.arrayContaining(["캐릭터 이름표는 역할 중심으로 압축", "하단 rail은 맵 바닥과 분리"]));
    expect(summaryPolish.characterLabelMode).toBe("minimal");
    expect(summaryPolish.lowerRailMode).toBe("detached");
    expect(`${standardPolish.stageLabel} ${standardPolish.notes.join(" ")} ${standardPolish.mapClassName}`).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

  it("builds Stage 12-A responsive readability plans from viewport width only", () => {
    const characters = Array.from({ length: 12 }, (_, index) => ({
      id: `character-${index}`,
      role: "model" as const,
      roomId: "sessions" as const,
      label: index === 11 ? "raw prompt transcript secret must not matter" : `모델 캐릭터 ${index + 1}`,
      status: "active" as const,
      detail: "raw task body script must not matter",
      redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
      x: 20,
      y: 20,
    }));

    const standardPlan = buildOfficeMapDensityPlan("standard", characters);
    const narrow = buildOfficeResponsiveReadabilityPlan(standardPlan, { viewportWidth: 430 });
    const desktop = buildOfficeResponsiveReadabilityPlan(standardPlan, { viewportWidth: 1280 });

    expect(narrow).toMatchObject({
      stageLabel: "Stage 12-A 반응형",
      viewportMode: "narrow",
      recommendedDensityMode: "summary",
      mapClassName: "office-map--responsive office-map--mobile-readable",
      railClassName: "office-map-rail--mobile-stack",
    });
    expect(narrow.notes).toEqual(expect.arrayContaining(["좁은 화면에서는 요약 모드 권장", "맵 rail은 세로 흐름으로 읽힘"]));
    expect(desktop.viewportMode).toBe("desktop");
    expect(desktop.recommendedDensityMode).toBe("standard");
    expect(`${narrow.stageLabel} ${narrow.notes.join(" ")} ${narrow.mapClassName}`).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

  it("builds Stage 10-H keyboard jump targets with density-aware recent rail labels", () => {
    const characters = Array.from({ length: 9 }, (_, index) => ({
      id: `character-${index}`,
      role: "worker" as const,
      roomId: "work" as const,
      label: index === 8 ? "raw prompt transcript secret must not matter" : `작업자 ${index + 1}`,
      status: "working" as const,
      detail: "raw task body script must not matter",
      redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
      x: 20,
      y: 20,
    }));

    const standardTargets = buildOfficeMapJumpTargets(buildOfficeMapDensityPlan("standard", characters));
    const summaryTargets = buildOfficeMapJumpTargets(buildOfficeMapDensityPlan("summary", characters));

    expect(standardTargets.map((target) => [target.id, target.label, target.targetId, target.enabled])).toEqual([
      ["map", "지도", "office-map-canvas", true],
      ["usability", "사용성", "office-map-usability", true],
      ["recent", "최근 변화", "office-map-recent", true],
      ["inspector", "안전 정보", "office-safe-inspector", true],
    ]);
    expect(summaryTargets.find((target) => target.id === "recent")).toMatchObject({
      label: "최근 변화 접힘",
      targetId: "office-map-recent-collapsed",
      enabled: true,
    });
    expect(summaryTargets.map((target) => `${target.label} ${target.detail} ${target.targetId}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

  it("adapts safe RPG characters into current scene markers without exposing raw data", () => {
    const characters = buildOfficeCharacters(
      officeFixture({
        data_sources: [
          { id: "sessions", status: "ok", checked_at: "2026-05-09T00:00:00Z", item_count: 1 },
          { id: "cron", status: "partial", checked_at: "2026-05-09T00:00:00Z", item_count: 1 },
        ],
        agents: [{ id: "agent-1", model: "raw-model-name must not matter", status: "active", prompt: "raw prompt must not matter" }],
        automations: [{ id: "job-1", state: "scheduled", last_status: "ok", script: "raw script must not matter" }],
      }),
      buildOfficeMapNodes(officeFixture()),
    );

    const sceneObjects = buildOfficeCharacterSceneObjects(characters);

    expect(sceneObjects.map((object) => object.kind)).toEqual(expect.arrayContaining(["avatar", "machine"]));
    expect(sceneObjects.find((object) => object.kind === "avatar")?.label).toBe("모델 캐릭터 1");
    expect(sceneObjects.every((object) => object.id.startsWith("character:"))).toBe(true);
    expect(sceneObjects.map((object) => `${object.label} ${object.detail}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret|model-name/i);
  });

  it("builds Korean empty-source copy without exposing raw adapter data", () => {
    const plan = buildOfficeEmptySourceCopyPlan(
      officeFixture({
        data_sources: [],
        agents: [{ id: "agent-raw", prompt: "raw prompt must not matter", token: "sk-test-must-not-matter" }],
        work_items: [{ id: "task-raw", body: "raw task body must not matter" }],
      }),
    );

    expect(plan.title).toBe("아직 연결된 소스가 없습니다");
    expect(plan.detail).toContain("대시보드 오류가 아니라 안전 DTO가 비어 있는 상태");
    expect(plan.items.map((item) => item.label)).toEqual(["연결 상태", "읽기 범위", "다음 확인"]);
    expect(plan.items.map((item) => item.detail).join(" ")).toContain("미보고 소스 5개");
    expect(plan.items.map((item) => item.detail).join(" ")).toContain("읽기 전용");
    expect(`${plan.title} ${plan.detail} ${plan.items.map((item) => `${item.label} ${item.detail}`).join(" ")}`).not.toMatch(/raw|prompt|task_body|body|script|secret|token|sk-/i);
  });

  it("summarizes source health and empty fixture hints without leaking raw adapter errors", () => {
    const state = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-09T00:00:00Z", item_count: 2, warning_count: 0 },
        { id: "kanban", status: "partial", checked_at: "2026-05-09T00:00:00Z", item_count: 1, warning_count: 2, error_summary: "raw stack trace must not matter" },
        { id: "cron", status: "error", checked_at: "2026-05-09T00:00:00Z", item_count: 0, warning_count: 1, error_summary: "raw token must not matter" },
        { id: "topics", status: "unavailable", checked_at: "2026-05-09T00:00:00Z", item_count: 0, warning_count: 0 },
      ],
    });

    const summary = buildOfficeSourceHealthSummary(state);
    const hints = buildOfficeEmptyStateHints();

    expect(summary.counts).toEqual({ ok: 1, partial: 1, missing: 1, unavailable: 1, error: 1 });
    expect(summary.totalWarningCount).toBe(3);
    expect(summary.label).toBe("주의 필요");
    expect(summary.detail).toBe("정상 1 · 주의 2 · 공백/미연결 2 · 경고 3");
    expect(summary.missingSourceIds).toEqual(["provenance"]);
    expect(hints).toMatchObject({
      rooms: "방 투영이 없습니다. 외부 작업이 비었다는 뜻은 아니며, 연결된 안전 DTO만 기준으로 표시합니다.",
      agents: "세션 어댑터가 안전 메타데이터를 제공하지 않았습니다. 제목/미리보기 원문은 계속 숨깁니다.",
      automations: "cron 스타일 작업이 보고되지 않았습니다. 실행/일시정지 제어는 제공하지 않습니다.",
    });
    expect(`${summary.label} ${summary.detail} ${summary.missingSourceIds.join(" ")} ${Object.values(hints).join(" ")}`).not.toMatch(/raw|stack|token|prompt|transcript|body|script|secret/i);
  });

  it("keeps the empty office map resilient and explicit about source gaps", () => {
    const state = officeFixture();
    const nodes = buildOfficeMapNodes(state);
    const flows = buildOfficeMapFlows(nodes);
    const objects = buildOfficeSceneObjects(state, nodes);

    expect(nodes).toHaveLength(4);
    expect(nodes.map((node) => [node.id, node.count, node.health])).toEqual([
      ["sessions", 0, "missing"],
      ["work", 0, "missing"],
      ["automation", 0, "missing"],
      ["routing", 0, "missing"],
    ]);
    expect(flows.map((flow) => flow.health)).toEqual(["missing", "missing", "missing"]);
    expect(objects).toContainEqual(expect.objectContaining({ id: "routing-empty", label: "미연결 보관함", kind: "mail" }));
    expect(objects.map((object) => `${object.label} ${object.detail}`).join(" ")).not.toMatch(/raw|prompt|transcript|body|script|secret/i);
  });

  it("builds safe browser-local office state deltas for room counts, health, and attention", () => {
    const previous = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
        { id: "kanban", status: "ok", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
        { id: "cron", status: "error", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
        { id: "topics", status: "missing", checked_at: "2026-05-08T00:00:00Z", item_count: 0 },
      ],
      agents: [{ id: "session-1", source_platform: "cli", status: "active", prompt: "raw prompt must not matter" }],
      work_items: [{ id: "task-1", title: "Safe task", status: "open", body: "raw task body must not matter" }],
      automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", last_status: "ok", script: "raw script must not matter" }],
    });
    const next = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-08T00:01:00Z", item_count: 3 },
        { id: "kanban", status: "partial", checked_at: "2026-05-08T00:01:00Z", item_count: 0 },
        { id: "cron", status: "ok", checked_at: "2026-05-08T00:01:00Z", item_count: 2 },
        { id: "topics", status: "missing", checked_at: "2026-05-08T00:01:00Z", item_count: 0 },
      ],
      agents: Array.from({ length: 3 }, (_, index) => ({ id: `session-${index}`, source_platform: "cli", status: "active", prompt: "raw prompt must not matter" })),
      work_items: [{ id: "task-1", title: "Safe task", status: "blocked", body: "raw task body must not matter" }],
      automations: [
        { id: "job-1", name: "Cron job job-1", state: "scheduled", last_status: "ok", script: "raw script must not matter" },
        { id: "job-2", name: "Cron job job-2", state: "scheduled", last_status: "ok", script: "raw script must not matter" },
      ],
    });

    const delta = buildOfficeStateDelta(previous, next);

    expect(delta.nodeBadges).toMatchObject({
      sessions: [{ label: "+2", tone: "positive" }],
      work: [{ label: "상태 변경", tone: "warning" }],
      automation: [{ label: "+1", tone: "positive" }, { label: "상태 변경", tone: "positive" }],
      routing: [],
    });
    expect(delta.recentChanges.map((change) => change.label)).toEqual([
      "세션 +2",
      "작업 상태 정상 → 부분 연결",
      "자동화 +1",
      "자동화 상태 오류 → 정상",
      "흐름 세션에서 작업으로 변경",
      "흐름 작업에서 자동화로 변경",
      "흐름 자동화에서 라우팅으로 변경",
      "확인 필요 1 → 2",
    ]);
    expect(delta.recentChanges.map((change) => change.detail).join(" ")).not.toMatch(/raw|prompt|body|script|secret/i);
  });

  it("builds flow-level change hints from safe endpoint count and health deltas", () => {
    const previous = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
        { id: "kanban", status: "ok", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
        { id: "cron", status: "partial", checked_at: "2026-05-08T00:00:00Z", item_count: 1 },
      ],
      agents: [{ id: "session-1", source_platform: "cli", status: "active", prompt: "raw prompt must not matter" }],
      work_items: [{ id: "task-1", title: "Safe task", status: "open", body: "raw task body must not matter" }],
      automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", last_status: "ok", script: "raw script must not matter" }],
    });
    const next = officeFixture({
      data_sources: [
        { id: "sessions", status: "ok", checked_at: "2026-05-08T00:01:00Z", item_count: 2 },
        { id: "kanban", status: "partial", checked_at: "2026-05-08T00:01:00Z", item_count: 1 },
        { id: "cron", status: "ok", checked_at: "2026-05-08T00:01:00Z", item_count: 1 },
      ],
      agents: [
        { id: "session-1", source_platform: "cli", status: "active", prompt: "raw prompt must not matter" },
        { id: "session-2", source_platform: "telegram", status: "idle", prompt: "raw prompt must not matter" },
      ],
      work_items: [{ id: "task-1", title: "Safe task", status: "open", body: "raw task body must not matter" }],
      automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", last_status: "ok", script: "raw script must not matter" }],
    });

    const delta = buildOfficeStateDelta(previous, next);

    expect(delta.changedFlows).toEqual([
      { from: "sessions", to: "work", label: "세션에서 작업으로", tone: "warning" },
      { from: "work", to: "automation", label: "작업에서 자동화로", tone: "neutral" },
      { from: "automation", to: "routing", label: "자동화에서 라우팅으로", tone: "positive" },
    ]);
    expect(delta.recentChanges.map((change) => change.label)).toContain("흐름 세션에서 작업으로 변경");
    expect(delta.recentChanges.map((change) => change.detail).join(" ")).not.toMatch(/raw|prompt|body|script|secret/i);
  });

  it("builds safe automation next-run timing buckets without exposing cron bodies", () => {
    const summary = buildOfficeAutomationTimingSummary(
      officeFixture({
        automations: [
          { id: "overdue", next_run_at: "2026-05-08T23:55:00Z", prompt: "raw prompt must not matter" },
          { id: "soon", next_run_at: "2026-05-09T00:10:00Z", script: "raw script must not matter" },
          { id: "hour", next_run_at: "2026-05-09T00:45:00Z", body: "raw body must not matter" },
          { id: "today", next_run_at: "2026-05-09T12:00:00Z", secret: "raw secret must not matter" },
          { id: "later", next_run_at: "2026-05-10T00:05:00Z" },
          { id: "unknown", next_run_at: "not-a-date" },
        ],
      }),
      "2026-05-09T00:00:00Z",
    );

    expect(summary.counts).toEqual({ overdue: 1, under15m: 1, under1h: 1, today: 1, later: 1, unknown: 1 });
    expect(summary.primaryBucket).toEqual({ id: "overdue", label: "기한 지남", tone: "warning" });
    expect(`${summary.primaryBucket.label} ${Object.keys(summary.counts).join(" ")}`).not.toMatch(/raw|prompt|body|script|secret/i);
  });

  it("adds safe automation next-run bucket deltas to the recent rail", () => {
    const previous = officeFixture({
      automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", next_run_at: "2026-05-09T12:00:00Z", script: "raw script must not matter" }],
    });
    const next = officeFixture({
      automations: [{ id: "job-1", name: "Cron job job-1", state: "scheduled", next_run_at: "2026-05-09T00:45:00Z", prompt: "raw prompt must not matter" }],
    });

    const delta = buildOfficeStateDelta(previous, next, { now: "2026-05-09T00:00:00Z" });

    expect(delta.nodeBadges.automation).toContainEqual({ label: "일정 변경", tone: "warning" });
    expect(delta.recentChanges).toContainEqual({
      id: "automation:next-run-bucket:today->under1h",
      label: "자동화 다음 실행 오늘 → <1h",
      detail: "next_run_at 시간대만 비교 · 프롬프트/스크립트 제외",
      tone: "warning",
    });
    expect(delta.recentChanges.map((change) => `${change.label} ${change.detail}`).join(" ")).not.toMatch(/raw|prompt|body|script|secret/i);
  });

  it("resolves local live tracking interval from tab visibility and consecutive failures", () => {
    expect(resolveOfficeLiveTrackingInterval({ isVisible: true, consecutiveFailures: 0 })).toBe(30_000);
    expect(resolveOfficeLiveTrackingInterval({ isVisible: false, consecutiveFailures: 0 })).toBe(60_000);
    expect(resolveOfficeLiveTrackingInterval({ isVisible: true, consecutiveFailures: 1 })).toBe(60_000);
    expect(resolveOfficeLiveTrackingInterval({ isVisible: true, consecutiveFailures: 2 })).toBe(120_000);
    expect(resolveOfficeLiveTrackingInterval({ isVisible: false, consecutiveFailures: 2 })).toBe(120_000);
  });

  it("collapses duplicate browser-memory recent changes before applying the ring-buffer limit", () => {
    const current = [
      { id: "sessions:count:1->2", label: "세션 +1", detail: "안전 개수 1 → 2", tone: "positive" as const },
      { id: "work:health:ok->partial", label: "작업 상태 정상 → 부분 연결", detail: "상태만 비교", tone: "warning" as const },
    ];
    const incoming = [
      { id: "sessions:count:1->2", label: "세션 +1", detail: "안전 개수 1 → 2", tone: "positive" as const },
      { id: "automation:count:1->2", label: "자동화 +1", detail: "안전 개수 1 → 2", tone: "positive" as const },
    ];

    expect(mergeOfficeRecentChanges(incoming, current, 2).map((change) => change.id)).toEqual([
      "sessions:count:1->2",
      "automation:count:1->2",
    ]);
  });

  it("returns no dynamic badges or rail entries for the first snapshot", () => {
    const delta = buildOfficeStateDelta(null, officeFixture());

    expect(delta.hasChanges).toBe(false);
    expect(delta.recentChanges).toEqual([]);
    expect(delta.nodeBadges).toEqual({ sessions: [], work: [], automation: [], routing: [] });
    expect(delta.changedFlows).toEqual([]);
  });

});
