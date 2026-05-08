import type { OfficeState } from "@/lib/api";

export type AttentionItem = {
  id: string;
  label: string;
  detail: string;
};

export type OfficeMapNode = {
  id: "sessions" | "work" | "automation" | "routing";
  label: string;
  detail: string;
  zone: "entry" | "workbench" | "machine" | "routing";
  count: number;
  health: "ok" | "partial" | "missing" | "error";
  x: number;
  y: number;
};

export type OfficeMapFlow = {
  from: OfficeMapNode["id"];
  to: OfficeMapNode["id"];
  label: string;
  health: OfficeMapNode["health"];
};

export type OfficeSceneObject = {
  id: string;
  roomId: OfficeMapNode["id"];
  kind: "avatar" | "desk" | "machine" | "mail" | "alert";
  label: string;
  detail: string;
  health: OfficeMapNode["health"];
  x: number;
  y: number;
};

export type OfficeSceneObjectView = {
  glyph: string;
  title: string;
  toneClass: string;
  ariaHidden: true;
  interactive: false;
};

export function textField(row: Record<string, unknown>, key: string): string {
  const value = row[key];
  return typeof value === "string" && value.length > 0 ? value : "—";
}

export function numberField(row: Record<string, unknown>, key: string): number | null {
  const value = row[key];
  return typeof value === "number" ? value : null;
}

export function groupByText(rows: Array<Record<string, unknown>>, key: string, fallback = "unknown") {
  return rows.reduce<Record<string, Array<Record<string, unknown>>>>((acc, row) => {
    const value = textField(row, key);
    const group = value === "—" ? fallback : value;
    acc[group] = acc[group] ?? [];
    acc[group].push(row);
    return acc;
  }, {});
}

export function visibleRows<T>(rows: T[], limit: number, expanded: boolean): T[] {
  return expanded ? rows : rows.slice(0, limit);
}

export function buildOfficeMapNodes(state: OfficeState): OfficeMapNode[] {
  const sourceStatus = (id: string): OfficeMapNode["health"] => {
    const status = state.data_sources.find((source) => source.id === id)?.status;
    if (status === "error") return "error";
    if (status === "partial" || status === "unavailable") return "partial";
    if (status === "ok") return "ok";
    return "missing";
  };

  const routingHealth: OfficeMapNode["health"] = state.topics.length > 0 || state.provenance.length > 0 ? "ok" : sourceStatus("topics");

  return [
    {
      id: "sessions",
      label: "세션",
      detail: "최근 안전 세션 메타데이터",
      zone: "entry",
      count: state.agents.length,
      health: sourceStatus("sessions"),
      x: 24,
      y: 30,
    },
    {
      id: "work",
      label: "작업",
      detail: "본문 없는 칸반/작업 카드",
      zone: "workbench",
      count: state.work_items.length,
      health: sourceStatus("kanban"),
      x: 70,
      y: 30,
    },
    {
      id: "automation",
      label: "자동화",
      detail: "읽기 전용 기계로 표시한 cron 작업",
      zone: "machine",
      count: state.automations.length,
      health: sourceStatus("cron"),
      x: 24,
      y: 67,
    },
    {
      id: "routing",
      label: "라우팅",
      detail: "토픽/출처 투영",
      zone: "routing",
      count: state.topics.length + state.provenance.length,
      health: routingHealth,
      x: 70,
      y: 67,
    },
  ];
}

export function buildOfficeMapFlows(nodes: OfficeMapNode[]): OfficeMapFlow[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const flowDefs: Array<Omit<OfficeMapFlow, "health">> = [
    { from: "sessions", to: "work", label: "세션에서 작업으로" },
    { from: "work", to: "automation", label: "작업에서 자동화로" },
    { from: "automation", to: "routing", label: "자동화에서 라우팅으로" },
  ];
  const severity: Record<OfficeMapNode["health"], number> = { ok: 0, missing: 1, partial: 2, error: 3 };
  const healthBySeverity: OfficeMapNode["health"][] = ["ok", "missing", "partial", "error"];

  return flowDefs.map((flow) => {
    const from = byId.get(flow.from);
    const to = byId.get(flow.to);
    const score = Math.max(severity[from?.health ?? "missing"], severity[to?.health ?? "missing"]);
    return { ...flow, health: healthBySeverity[score] };
  });
}

const SCENE_OBJECT_LIMIT = 6;

const SCENE_SLOTS: Record<OfficeMapNode["id"], Array<[number, number]>> = {
  sessions: [[17, 22], [24, 22], [31, 22], [17, 34], [24, 34], [31, 34]],
  work: [[63, 21], [70, 21], [77, 21], [63, 34], [70, 34], [77, 34]],
  automation: [[17, 58], [24, 58], [31, 58], [17, 68], [24, 68], [31, 68]],
  routing: [[63, 58], [70, 58], [77, 58], [63, 68], [70, 68], [77, 68]],
};

const SCENE_ROOM_CONFIG: Record<OfficeMapNode["id"], { kind: OfficeSceneObject["kind"]; singular: string; plural: string; emptyLabel?: string; emptyDetail?: string }> = {
  sessions: { kind: "avatar", singular: "세션 표시", plural: "세션" },
  work: { kind: "desk", singular: "작업 책상", plural: "작업" },
  automation: { kind: "machine", singular: "자동화 기계", plural: "자동화" },
  routing: { kind: "mail", singular: "라우팅 우편", plural: "경로", emptyLabel: "미연결 보관함", emptyDetail: "토픽/출처 공백을 명시" },
};

function roomRows(state: OfficeState, roomId: OfficeMapNode["id"]): Array<Record<string, unknown>> {
  if (roomId === "sessions") return state.agents;
  if (roomId === "work") return state.work_items;
  if (roomId === "automation") return state.automations;
  return [...state.topics, ...state.provenance];
}

function sceneObjectGlyph(kind: OfficeSceneObject["kind"]): string {
  if (kind === "avatar") return "●";
  if (kind === "desk") return "▤";
  if (kind === "machine") return "▣";
  if (kind === "mail") return "▥";
  return "!";
}

function sceneObjectTone(health: OfficeSceneObject["health"]): string {
  if (health === "ok") return "border-emerald-200/70 bg-emerald-300/25 text-emerald-50 shadow-emerald-950/40";
  if (health === "partial") return "border-yellow-200/80 bg-yellow-300/25 text-yellow-50 shadow-yellow-950/40";
  if (health === "error") return "border-red-200/80 bg-red-300/25 text-red-50 shadow-red-950/40";
  return "border-sky-200/65 bg-sky-300/20 text-sky-50 shadow-sky-950/35";
}

export function buildOfficeSceneObjectView(object: OfficeSceneObject): OfficeSceneObjectView {
  return {
    glyph: sceneObjectGlyph(object.kind),
    title: `${object.label} · ${object.detail}`,
    toneClass: sceneObjectTone(object.health),
    ariaHidden: true,
    interactive: false,
  };
}

export function buildOfficeSceneObjects(state: OfficeState, nodes: OfficeMapNode[]): OfficeSceneObject[] {
  return nodes.flatMap((node) => {
    const config = SCENE_ROOM_CONFIG[node.id];
    const rows = roomRows(state, node.id);
    const slots = SCENE_SLOTS[node.id];
    const visibleRows = rows.slice(0, SCENE_OBJECT_LIMIT);
    const objects = visibleRows.map<OfficeSceneObject>((_, index) => {
      const [x, y] = slots[index];
      return {
        id: `${node.id}-${config.kind}-${index + 1}`,
        roomId: node.id,
        kind: config.kind,
        label: `${config.singular} ${index + 1}`,
        detail: `${node.zone} 안전 표시`,
        health: node.health,
        x,
        y,
      };
    });

    if (rows.length === 0 && config.emptyLabel) {
      const [x, y] = slots[0];
      objects.push({
        id: `${node.id}-empty`,
        roomId: node.id,
        kind: config.kind,
        label: config.emptyLabel,
        detail: config.emptyDetail ?? `${node.zone} 빈 표시`,
        health: node.health,
        x,
        y,
      });
    }

    if (rows.length > SCENE_OBJECT_LIMIT) {
      objects.push({
        id: `${node.id}-overflow`,
        roomId: node.id,
        kind: "alert",
        label: `+${rows.length - SCENE_OBJECT_LIMIT} ${config.plural}`,
        detail: "지도 밀도 때문에 숨긴 추가 안전 개수",
        health: node.health,
        x: Math.min(node.x + 12, 90),
        y: Math.min(node.y + 11, 88),
      });
    }

    return objects;
  });
}

export function buildOfficeAttentionItems(state: OfficeState): AttentionItem[] {
  const blocked = state.work_items
    .map((item) => ({
      id: `work:${String(item.id)}`,
      label: textField(item, "title"),
      detail: `작업 항목 · ${textField(item, "status")}`,
    }))
    .filter((item) => item.detail.includes("blocked"));
  const failedAutomations = state.automations
    .filter(
      (job) => job.last_status === "error" || job.state === "error" || (Array.isArray(job.badges) && job.badges.includes("needs_attention")),
    )
    .map((job) => ({
      id: `automation:${String(job.id)}`,
      label: textField(job, "name"),
      detail: `자동화 · ${textField(job, "last_status")}`,
    }));
  const sourceWarnings = state.data_sources
    .filter((source) => source.status === "partial" || source.status === "error" || (source.warning_count ?? 0) > 0)
    .map((source) => ({
      id: `source:${source.id}`,
      label: source.id,
      detail: `source · ${source.status}${source.warning_count ? ` · ${source.warning_count} warning(s)` : ""}`,
    }));
  return [...blocked, ...failedAutomations, ...sourceWarnings];
}
