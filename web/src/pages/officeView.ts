import type { OfficeSourceStatus, OfficeState } from "@/lib/api";

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

export type OfficeSceneMotionStyle = Record<"--office-motion-x" | "--office-motion-y" | "--office-motion-duration" | "--office-motion-delay", string>;

export type OfficeSceneMotionTrack = {
  className: string;
  style: OfficeSceneMotionStyle;
  ariaLabel: string;
};

export type OfficeCharacterRole = "model" | "operator" | "worker" | "reviewer" | "automation_keeper" | "router" | "sentinel" | "alert";

export type OfficeCharacterStatus = "active" | "idle" | "working" | "reviewing" | "routing" | "scheduled" | "blocked" | "warning" | "error" | "unknown";

export type OfficeCharacter = {
  id: string;
  role: OfficeCharacterRole;
  roomId: OfficeMapNode["id"];
  label: string;
  status: OfficeCharacterStatus;
  detail: string;
  redactionNote: string;
  x: number;
  y: number;
};

export type OfficeCharacterView = {
  glyph: string;
  bodyClassName: string;
  accessoryClassName: string;
  nameplate: string;
  statusLabel: string;
  safeTitle: string;
};

export type OfficeCharacterActivityId = "thinking" | "working" | "reviewing" | "routing" | "scheduled" | "soon" | "blocked" | "warning" | "idle" | "unknown";

export type OfficeCharacterActivity = {
  id: OfficeCharacterActivityId;
  label: string;
  motion: "walk" | "idle" | "blink" | "pulse" | "none";
  tone: "normal" | "success" | "warning" | "danger" | "muted";
  reducedMotionLabel: string;
};

export type OfficeCharacterRoute = {
  id: string;
  fromRoomId: OfficeMapNode["id"];
  toRoomId: OfficeMapNode["id"];
  label: string;
  detail: string;
  tone: "normal" | "warning" | "danger";
  motion: "route" | "alert";
  changed: true;
  reducedMotionLabel: string;
};

export type OfficeCharacterInspector = {
  kind: "RPG 캐릭터";
  title: string;
  ariaLabel: string;
  fields: Array<[string, string]>;
};

export type OfficeDeltaBadge = {
  label: string;
  tone: "positive" | "negative" | "warning" | "neutral";
};

export type OfficeRecentChange = {
  id: string;
  label: string;
  detail: string;
  tone: OfficeDeltaBadge["tone"];
};

export type OfficeFlowChange = {
  from: OfficeMapFlow["from"];
  to: OfficeMapFlow["to"];
  label: string;
  tone: OfficeDeltaBadge["tone"];
};

export type OfficeAutomationTimingBucketId = "overdue" | "under15m" | "under1h" | "today" | "later" | "unknown";

export type OfficeAutomationTimingBucket = {
  id: OfficeAutomationTimingBucketId;
  label: string;
  tone: OfficeDeltaBadge["tone"];
};

export type OfficeAutomationTimingSummary = {
  counts: Record<OfficeAutomationTimingBucketId, number>;
  primaryBucket: OfficeAutomationTimingBucket;
};

export type OfficeStateDelta = {
  hasChanges: boolean;
  nodeBadges: Record<OfficeMapNode["id"], OfficeDeltaBadge[]>;
  changedFlows: OfficeFlowChange[];
  recentChanges: OfficeRecentChange[];
};

export type OfficeSourceHealthSummary = {
  counts: Record<OfficeSourceStatus, number>;
  label: string;
  detail: string;
  totalWarningCount: number;
  missingSourceIds: string[];
  tone: OfficeDeltaBadge["tone"];
};

export type OfficeEmptySourceCopyItem = {
  label: string;
  detail: string;
  tone: OfficeDeltaBadge["tone"];
};

export type OfficeEmptySourceCopyPlan = {
  title: string;
  detail: string;
  items: OfficeEmptySourceCopyItem[];
};

export type OfficeEmptyStateHints = Record<"rooms" | "agents" | "workItems" | "automations" | "topics" | "events", string>;

export type OfficeUsabilityItem = {
  id: "density" | "source-fallback" | "motion" | "responsive" | "korean-copy";
  label: string;
  detail: string;
  tone: OfficeDeltaBadge["tone"];
};

export type OfficeUsabilitySummary = {
  items: OfficeUsabilityItem[];
};

export type OfficeMapDensityMode = "summary" | "standard" | "detail";

export type OfficeMapDensityPlan = {
  mode: OfficeMapDensityMode;
  label: string;
  detail: string;
  visibleCharacters: OfficeCharacter[];
  hiddenCharacterCount: number;
  showUsabilityRail: boolean;
  showRecentRail: boolean;
};

export type OfficeMapJumpTarget = {
  id: "map" | "usability" | "recent" | "inspector";
  label: string;
  detail: string;
  targetId: string;
  enabled: boolean;
};

export type OfficeMapPolishPlan = {
  stageLabel: string;
  characterLabelMode: "minimal" | "compact" | "full";
  lowerRailMode: "inline" | "detached";
  mapClassName: string;
  legendClassName: string;
  notes: string[];
};

export type OfficeResponsiveReadabilityPlan = {
  stageLabel: string;
  viewportMode: "narrow" | "desktop";
  recommendedDensityMode: OfficeMapDensityMode;
  mapClassName: string;
  railClassName: string;
  notes: string[];
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

const EXPECTED_OFFICE_SOURCE_IDS = ["sessions", "kanban", "cron", "topics", "provenance"] as const;

export function buildOfficeSourceHealthSummary(state: OfficeState): OfficeSourceHealthSummary {
  const counts: Record<OfficeSourceStatus, number> = { ok: 0, partial: 0, missing: 0, unavailable: 0, error: 0 };
  const seen = new Set<string>();
  let totalWarningCount = 0;

  for (const source of state.data_sources) {
    counts[source.status] += 1;
    totalWarningCount += source.warning_count ?? 0;
    seen.add(source.id);
  }

  const missingSourceIds = EXPECTED_OFFICE_SOURCE_IDS.filter((id) => !seen.has(id));
  counts.missing += missingSourceIds.length;

  const attentionCount = counts.partial + counts.error;
  const gapCount = counts.missing + counts.unavailable;
  const label = counts.error > 0 || totalWarningCount > 0 || counts.partial > 0 ? "주의 필요" : gapCount > 0 ? "소스 공백" : "정상";
  const tone: OfficeDeltaBadge["tone"] = label === "정상" ? "positive" : label === "주의 필요" ? "warning" : "neutral";

  return {
    counts,
    label,
    detail: `정상 ${counts.ok} · 주의 ${attentionCount} · 공백/미연결 ${gapCount} · 경고 ${totalWarningCount}`,
    totalWarningCount,
    missingSourceIds,
    tone,
  };
}

export function buildOfficeEmptySourceCopyPlan(state: OfficeState): OfficeEmptySourceCopyPlan {
  const sourceHealth = buildOfficeSourceHealthSummary(state);
  const missingCount = sourceHealth.missingSourceIds.length;
  const reportedCount = state.data_sources.length;
  const gapCount = sourceHealth.counts.missing + sourceHealth.counts.unavailable;

  return {
    title: reportedCount === 0 ? "아직 연결된 소스가 없습니다" : "일부 소스가 비어 있습니다",
    detail:
      reportedCount === 0
        ? "대시보드 오류가 아니라 안전 DTO가 비어 있는 상태입니다. 연결 전에도 읽기 전용 셸과 가림 정책은 유지됩니다."
        : "보고된 소스와 미보고 소스를 분리해 표시합니다. 비어 있는 영역은 민감 원문을 추론하지 않습니다.",
    items: [
      {
        label: "연결 상태",
        detail: missingCount > 0 ? `미보고 소스 ${missingCount}개 · ${sourceHealth.missingSourceIds.join(" · ")}` : `보고 소스 ${reportedCount}개`,
        tone: gapCount > 0 ? "neutral" : "positive",
      },
      {
        label: "읽기 범위",
        detail: "읽기 전용 안전 DTO의 개수·상태·시각만 사용합니다.",
        tone: "positive",
      },
      {
        label: "다음 확인",
        detail: "필요하면 어댑터 연결 상태를 확인하되 이 화면에서는 실행·수정 제어를 제공하지 않습니다.",
        tone: "neutral",
      },
    ],
  };
}

export function buildOfficeEmptyStateHints(): OfficeEmptyStateHints {
  return {
    rooms: "방 투영이 없습니다. 외부 작업이 비었다는 뜻은 아니며, 연결된 안전 DTO만 기준으로 표시합니다.",
    agents: "세션 어댑터가 안전 메타데이터를 제공하지 않았습니다. 제목/미리보기 원문은 계속 숨깁니다.",
    workItems: "승인된 어댑터가 작업 카드를 보고하지 않았습니다. 작업 본문/결과/댓글/로그는 계속 제외합니다.",
    automations: "cron 스타일 작업이 보고되지 않았습니다. 실행/일시정지 제어는 제공하지 않습니다.",
    topics: "토픽 레지스트리/투영이 연결되어 있지 않습니다. UI 오류가 아니라 알려진 소스 공백일 수 있습니다.",
    events: "안전 시간표가 생성되지 않았습니다. 원문 로그와 대화 기록은 설계상 숨깁니다.",
  };
}

export function buildOfficeUsabilitySummary(
  state: OfficeState,
  characters: OfficeCharacter[],
  options: { reducedMotion?: boolean; viewportWidth?: number } = {},
): OfficeUsabilitySummary {
  const sourceHealth = buildOfficeSourceHealthSummary(state);
  const hasDenseCharacters = characters.length >= 12 || characters.some((character) => character.label.startsWith("+"));
  const fallbackCount = sourceHealth.counts.missing + sourceHealth.counts.unavailable + sourceHealth.counts.partial;
  const isNarrow = typeof options.viewportWidth === "number" && options.viewportWidth < 640;

  return {
    items: [
      {
        id: "density",
        label: "밀도 점검",
        detail: hasDenseCharacters ? "합산 캐릭터와 +N 표식으로 겹침을 줄입니다." : "현재 캐릭터 밀도는 안정적입니다.",
        tone: hasDenseCharacters ? "warning" : "positive",
      },
      {
        id: "source-fallback",
        label: "소스 공백",
        detail: fallbackCount > 0 ? `미연결/부분 연결 소스 ${fallbackCount}개를 빈 방·보관함으로 표시합니다.` : "보고된 소스가 정상 범위입니다.",
        tone: fallbackCount > 0 ? "neutral" : "positive",
      },
      {
        id: "motion",
        label: "동작 모드",
        detail: options.reducedMotion ? "사용자 설정에 따라 애니메이션은 정지하고 정적 라벨을 유지합니다." : "CSS 전용 약한 움직임을 사용하며 원문 데이터는 쓰지 않습니다.",
        tone: "neutral",
      },
      {
        id: "responsive",
        label: isNarrow ? "좁은 화면" : "화면 배치",
        detail: isNarrow ? "좁은 화면에서는 맵·최근 변화·안전 정보가 세로 흐름으로 읽힙니다." : "데스크톱에서는 맵과 안전 정보가 분리되어 읽힙니다.",
        tone: "neutral",
      },
      {
        id: "korean-copy",
        label: "한국어 우선",
        detail: "주요 안내·범례·안전 문구는 한국어로 유지하고 안정 식별자만 그대로 둡니다.",
        tone: "positive",
      },
    ],
  };
}

const OFFICE_DENSITY_MODE_CONFIG: Record<OfficeMapDensityMode, { label: string; limit: number; showUsabilityRail: boolean; showRecentRail: boolean }> = {
  summary: { label: "요약", limit: 6, showUsabilityRail: true, showRecentRail: false },
  standard: { label: "표준", limit: 12, showUsabilityRail: true, showRecentRail: true },
  detail: { label: "상세", limit: Number.POSITIVE_INFINITY, showUsabilityRail: true, showRecentRail: true },
};

export function buildOfficeMapDensityPlan(mode: OfficeMapDensityMode, characters: OfficeCharacter[]): OfficeMapDensityPlan {
  const config = OFFICE_DENSITY_MODE_CONFIG[mode] ?? OFFICE_DENSITY_MODE_CONFIG.standard;
  const visibleCharacters = characters.slice(0, config.limit);
  const hiddenCharacterCount = Math.max(characters.length - visibleCharacters.length, 0);
  const detail =
    hiddenCharacterCount > 0
      ? `${config.label} 모드 · 캐릭터 ${visibleCharacters.length}개 표시 · ${hiddenCharacterCount}개는 안전하게 접음`
      : `${config.label} 모드 · 캐릭터 ${visibleCharacters.length}개 표시`;
  return {
    mode,
    label: config.label,
    detail,
    visibleCharacters,
    hiddenCharacterCount,
    showUsabilityRail: config.showUsabilityRail,
    showRecentRail: config.showRecentRail,
  };
}

export function buildOfficeMapJumpTargets(densityPlan: OfficeMapDensityPlan): OfficeMapJumpTarget[] {
  return [
    { id: "map", label: "지도", detail: "오피스 맵 방과 캐릭터로 이동", targetId: "office-map-canvas", enabled: true },
    { id: "usability", label: "사용성", detail: "밀도·소스·동작 점검 rail로 이동", targetId: "office-map-usability", enabled: densityPlan.showUsabilityRail },
    {
      id: "recent",
      label: densityPlan.showRecentRail ? "최근 변화" : "최근 변화 접힘",
      detail: densityPlan.showRecentRail ? "브라우저 메모리의 안전 delta rail로 이동" : "요약 모드에서 접힌 최근 변화 안내로 이동",
      targetId: densityPlan.showRecentRail ? "office-map-recent" : "office-map-recent-collapsed",
      enabled: true,
    },
    { id: "inspector", label: "안전 정보", detail: "선택한 방 또는 캐릭터의 안전 정보 패널로 이동", targetId: "office-safe-inspector", enabled: true },
  ];
}

export function buildOfficeMapPolishPlan(densityPlan: OfficeMapDensityPlan): OfficeMapPolishPlan {
  const visibleCount = densityPlan.visibleCharacters.length;
  const characterLabelMode: OfficeMapPolishPlan["characterLabelMode"] =
    densityPlan.mode === "summary" ? "minimal" : visibleCount >= 10 || densityPlan.hiddenCharacterCount > 0 ? "compact" : "full";
  const lowerRailMode: OfficeMapPolishPlan["lowerRailMode"] = visibleCount >= 6 || densityPlan.showRecentRail ? "detached" : "inline";
  const mapClasses = ["office-map--polished"];
  if (characterLabelMode === "minimal") mapClasses.push("office-map--labels-minimal");
  if (characterLabelMode === "compact") mapClasses.push("office-map--labels-compact");
  if (lowerRailMode === "detached") mapClasses.push("office-map--rail-detached");
  return {
    stageLabel: "Stage 11-B 정돈",
    characterLabelMode,
    lowerRailMode,
    mapClassName: mapClasses.join(" "),
    legendClassName: `office-map-legend${lowerRailMode === "detached" ? " office-map-legend--detached" : ""}`,
    notes: [
      characterLabelMode === "full" ? "캐릭터 이름표는 전체 표시" : "캐릭터 이름표는 역할 중심으로 압축",
      lowerRailMode === "detached" ? "하단 rail은 맵 바닥과 분리" : "하단 rail은 맵 안에서 유지",
    ],
  };
}

export function buildOfficeResponsiveReadabilityPlan(
  densityPlan: OfficeMapDensityPlan,
  options: { viewportWidth?: number } = {},
): OfficeResponsiveReadabilityPlan {
  const isNarrow = typeof options.viewportWidth === "number" && options.viewportWidth < 640;
  return {
    stageLabel: "Stage 12-A 반응형",
    viewportMode: isNarrow ? "narrow" : "desktop",
    recommendedDensityMode: isNarrow ? "summary" : densityPlan.mode,
    mapClassName: `office-map--responsive${isNarrow ? " office-map--mobile-readable" : ""}`,
    railClassName: isNarrow ? "office-map-rail--mobile-stack" : "office-map-rail--desktop",
    notes: isNarrow
      ? ["좁은 화면에서는 요약 모드 권장", "맵 rail은 세로 흐름으로 읽힘"]
      : ["데스크톱에서는 현재 밀도 모드 유지", "맵과 rail은 분리된 영역으로 읽힘"],
  };
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

const DELTA_HEALTH_LABEL: Record<OfficeMapNode["health"], string> = {
  ok: "정상",
  partial: "부분 연결",
  missing: "미연결",
  error: "오류",
};

const AUTOMATION_TIMING_BUCKETS: Record<OfficeAutomationTimingBucketId, OfficeAutomationTimingBucket> = {
  overdue: { id: "overdue", label: "기한 지남", tone: "warning" },
  under15m: { id: "under15m", label: "<15m", tone: "warning" },
  under1h: { id: "under1h", label: "<1h", tone: "warning" },
  today: { id: "today", label: "오늘", tone: "neutral" },
  later: { id: "later", label: "나중", tone: "neutral" },
  unknown: { id: "unknown", label: "알 수 없음", tone: "neutral" },
};

const AUTOMATION_TIMING_PRIORITY: OfficeAutomationTimingBucketId[] = ["overdue", "under15m", "under1h", "today", "later", "unknown"];

export const OFFICE_LIVE_TRACKING_BASE_INTERVAL_MS = 30_000;

function nowMs(now: Date | string | number = Date.now()): number {
  if (typeof now === "number") return now;
  if (now instanceof Date) return now.getTime();
  return new Date(now).getTime();
}

function automationTimingBucket(nextRunAt: unknown, now: number): OfficeAutomationTimingBucketId {
  if (typeof nextRunAt !== "string" || nextRunAt.length === 0) return "unknown";
  const runAt = new Date(nextRunAt).getTime();
  if (Number.isNaN(runAt) || Number.isNaN(now)) return "unknown";
  const deltaMs = runAt - now;
  if (deltaMs < 0) return "overdue";
  if (deltaMs < 15 * 60 * 1000) return "under15m";
  if (deltaMs < 60 * 60 * 1000) return "under1h";
  const runDate = new Date(runAt);
  const nowDate = new Date(now);
  if (runDate.getUTCFullYear() === nowDate.getUTCFullYear() && runDate.getUTCMonth() === nowDate.getUTCMonth() && runDate.getUTCDate() === nowDate.getUTCDate()) return "today";
  return "later";
}

export function buildOfficeAutomationTimingSummary(state: OfficeState, now: Date | string | number = Date.now()): OfficeAutomationTimingSummary {
  const referenceTime = nowMs(now);
  const counts: Record<OfficeAutomationTimingBucketId, number> = { overdue: 0, under15m: 0, under1h: 0, today: 0, later: 0, unknown: 0 };
  for (const automation of state.automations) {
    counts[automationTimingBucket(automation.next_run_at, referenceTime)] += 1;
  }
  const primaryId = AUTOMATION_TIMING_PRIORITY.find((bucket) => counts[bucket] > 0) ?? "unknown";
  return { counts, primaryBucket: AUTOMATION_TIMING_BUCKETS[primaryId] };
}

function timingBucketTone(previous: OfficeAutomationTimingBucketId, next: OfficeAutomationTimingBucketId): OfficeDeltaBadge["tone"] {
  const urgency: Record<OfficeAutomationTimingBucketId, number> = { overdue: 5, under15m: 4, under1h: 3, today: 2, later: 1, unknown: 0 };
  if (urgency[next] > urgency[previous]) return "warning";
  if (urgency[next] < urgency[previous]) return "positive";
  return "neutral";
}

export function resolveOfficeLiveTrackingInterval({ isVisible, consecutiveFailures }: { isVisible: boolean; consecutiveFailures: number }): number {
  const visibilityDelay = isVisible ? OFFICE_LIVE_TRACKING_BASE_INTERVAL_MS : 60_000;
  const failureDelay = consecutiveFailures >= 2 ? 120_000 : consecutiveFailures === 1 ? 60_000 : OFFICE_LIVE_TRACKING_BASE_INTERVAL_MS;
  return Math.max(visibilityDelay, failureDelay);
}

function deltaTone(value: number): OfficeDeltaBadge["tone"] {
  if (value > 0) return "positive";
  if (value < 0) return "negative";
  return "neutral";
}

function healthChangeTone(previous: OfficeMapNode["health"], next: OfficeMapNode["health"]): OfficeDeltaBadge["tone"] {
  const severity: Record<OfficeMapNode["health"], number> = { ok: 0, missing: 1, partial: 2, error: 3 };
  if (severity[next] < severity[previous]) return "positive";
  if (severity[next] > severity[previous]) return "warning";
  return "neutral";
}

function flowEndpointTone(previousNodes: OfficeMapNode[], nextNodes: OfficeMapNode[]): OfficeDeltaBadge["tone"] {
  const previousSeverityTotal = previousNodes.reduce((total, node) => total + ({ ok: 0, missing: 1, partial: 2, error: 3 }[node.health]), 0);
  const nextSeverityTotal = nextNodes.reduce((total, node) => total + ({ ok: 0, missing: 1, partial: 2, error: 3 }[node.health]), 0);
  if (nextSeverityTotal < previousSeverityTotal) return "positive";
  if (nextSeverityTotal > previousSeverityTotal) return "warning";
  return "neutral";
}

function flowChanged(previousFlow: OfficeMapFlow, nextFlow: OfficeMapFlow, previousEndpoints: OfficeMapNode[], nextEndpoints: OfficeMapNode[]): boolean {
  return (
    previousFlow.health !== nextFlow.health ||
    previousEndpoints.some((node, index) => node.health !== nextEndpoints[index]?.health || node.count !== nextEndpoints[index]?.count)
  );
}

export function buildOfficeStateDelta(
  previous: OfficeState | null | undefined,
  next: OfficeState,
  options: { now?: Date | string | number } = {},
): OfficeStateDelta {
  const nodeBadges: Record<OfficeMapNode["id"], OfficeDeltaBadge[]> = {
    sessions: [],
    work: [],
    automation: [],
    routing: [],
  };
  const recentChanges: OfficeRecentChange[] = [];
  const changedFlows: OfficeFlowChange[] = [];
  if (!previous) {
    return { hasChanges: false, nodeBadges, changedFlows, recentChanges };
  }

  const previousNodes = buildOfficeMapNodes(previous);
  const nextNodes = buildOfficeMapNodes(next);
  const previousById = new Map(previousNodes.map((node) => [node.id, node]));
  const nextById = new Map(nextNodes.map((node) => [node.id, node]));
  const previousFlows = buildOfficeMapFlows(previousNodes);
  const nextFlows = buildOfficeMapFlows(nextNodes);
  const previousFlowById = new Map(previousFlows.map((flow) => [`${flow.from}->${flow.to}`, flow]));
  const previousAttentionCount = buildOfficeAttentionItems(previous).length;
  const nextAttentionCount = buildOfficeAttentionItems(next).length;
  const previousAutomationTiming = buildOfficeAutomationTimingSummary(previous, options.now);
  const nextAutomationTiming = buildOfficeAutomationTimingSummary(next, options.now);

  for (const node of nextNodes) {
    const before = previousById.get(node.id);
    if (!before) continue;
    const countDelta = node.count - before.count;
    if (countDelta !== 0) {
      const label = `${countDelta > 0 ? "+" : ""}${countDelta}`;
      const tone = deltaTone(countDelta);
      nodeBadges[node.id].push({ label, tone });
      recentChanges.push({
        id: `${node.id}:count:${before.count}->${node.count}`,
        label: `${node.label} ${label}`,
        detail: `안전 개수 ${before.count} → ${node.count}`,
        tone,
      });
    }
    if (node.health !== before.health) {
      const tone = healthChangeTone(before.health, node.health);
      nodeBadges[node.id].push({ label: "상태 변경", tone });
      recentChanges.push({
        id: `${node.id}:health:${before.health}->${node.health}`,
        label: `${node.label} 상태 ${DELTA_HEALTH_LABEL[before.health]} → ${DELTA_HEALTH_LABEL[node.health]}`,
        detail: `${node.detail} · 상태만 비교`,
        tone,
      });
    }
  }

  for (const flow of nextFlows) {
    const before = previousFlowById.get(`${flow.from}->${flow.to}`);
    const previousFrom = previousById.get(flow.from);
    const previousTo = previousById.get(flow.to);
    const nextFrom = nextById.get(flow.from);
    const nextTo = nextById.get(flow.to);
    if (!before || !previousFrom || !previousTo || !nextFrom || !nextTo) continue;
    const previousEndpoints = [previousFrom, previousTo];
    const nextEndpoints = [nextFrom, nextTo];
    if (!flowChanged(before, flow, previousEndpoints, nextEndpoints)) continue;
    const tone = before.health !== flow.health ? healthChangeTone(before.health, flow.health) : flowEndpointTone(previousEndpoints, nextEndpoints);
    changedFlows.push({ from: flow.from, to: flow.to, label: flow.label, tone });
    recentChanges.push({
      id: `flow:${flow.from}->${flow.to}:${before.health}->${flow.health}:${previousFrom.count},${previousTo.count}->${nextFrom.count},${nextTo.count}`,
      label: `흐름 ${flow.label} 변경`,
      detail: `흐름 상태 ${DELTA_HEALTH_LABEL[before.health]} → ${DELTA_HEALTH_LABEL[flow.health]} · endpoint 개수/상태만 비교`,
      tone,
    });
  }

  if (previousAutomationTiming.primaryBucket.id !== nextAutomationTiming.primaryBucket.id) {
    const tone = timingBucketTone(previousAutomationTiming.primaryBucket.id, nextAutomationTiming.primaryBucket.id);
    nodeBadges.automation.push({ label: "일정 변경", tone });
    recentChanges.push({
      id: `automation:next-run-bucket:${previousAutomationTiming.primaryBucket.id}->${nextAutomationTiming.primaryBucket.id}`,
      label: `자동화 다음 실행 ${previousAutomationTiming.primaryBucket.label} → ${nextAutomationTiming.primaryBucket.label}`,
      detail: "next_run_at 시간대만 비교 · 프롬프트/스크립트 제외",
      tone,
    });
  }

  if (previousAttentionCount !== nextAttentionCount) {
    recentChanges.push({
      id: `attention:${previousAttentionCount}->${nextAttentionCount}`,
      label: `확인 필요 ${previousAttentionCount} → ${nextAttentionCount}`,
      detail: "막힌 작업, 실패한 자동화, 소스 경고 개수 변화",
      tone: deltaTone(nextAttentionCount - previousAttentionCount),
    });
  }

  return { hasChanges: recentChanges.length > 0, nodeBadges, changedFlows, recentChanges };
}

export function mergeOfficeRecentChanges(incoming: OfficeRecentChange[], current: OfficeRecentChange[], limit: number): OfficeRecentChange[] {
  const merged: OfficeRecentChange[] = [];
  const seen = new Set<string>();
  for (const change of [...incoming, ...current]) {
    if (seen.has(change.id)) continue;
    seen.add(change.id);
    merged.push(change);
    if (merged.length >= limit) break;
  }
  return merged;
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

const CHARACTER_LIMIT_PER_ROLE = 3;

const CHARACTER_ACTIVITY_LABEL: Record<OfficeCharacterActivityId, string> = {
  thinking: "생각 중",
  working: "작업 중",
  reviewing: "검토 중",
  routing: "전달 중",
  scheduled: "예약 대기",
  soon: "곧 실행",
  blocked: "막힘",
  warning: "확인 필요",
  idle: "대기",
  unknown: "확인 불가",
};

const CHARACTER_ACTIVITY_STYLE: Record<OfficeCharacterActivityId, Pick<OfficeCharacterActivity, "motion" | "tone">> = {
  thinking: { motion: "idle", tone: "normal" },
  working: { motion: "pulse", tone: "success" },
  reviewing: { motion: "idle", tone: "success" },
  routing: { motion: "walk", tone: "normal" },
  scheduled: { motion: "idle", tone: "muted" },
  soon: { motion: "blink", tone: "warning" },
  blocked: { motion: "blink", tone: "danger" },
  warning: { motion: "blink", tone: "warning" },
  idle: { motion: "none", tone: "muted" },
  unknown: { motion: "none", tone: "muted" },
};

const CHARACTER_ROLE_LABEL: Record<OfficeCharacterRole, string> = {
  model: "모델 캐릭터",
  operator: "조작자",
  worker: "작업자",
  reviewer: "검토자",
  automation_keeper: "자동화 관리인",
  router: "전달자",
  sentinel: "감시자",
  alert: "경보 담당",
};

const CHARACTER_ROLE_NAMEPLATE: Record<OfficeCharacterRole, string> = {
  model: "모델",
  operator: "조작",
  worker: "작업",
  reviewer: "검토",
  automation_keeper: "자동화",
  router: "전달",
  sentinel: "감시",
  alert: "경보",
};

const CHARACTER_ROLE_GLYPH: Record<OfficeCharacterRole, string> = {
  model: "◇",
  operator: "◆",
  worker: "▤",
  reviewer: "◎",
  automation_keeper: "▣",
  router: "✉",
  sentinel: "◈",
  alert: "!",
};

const CHARACTER_STATUS_LABEL: Record<OfficeCharacterStatus, string> = {
  active: "활성",
  idle: "대기",
  working: "활성",
  reviewing: "검토",
  routing: "전달",
  scheduled: "예약",
  blocked: "막힘",
  warning: "주의",
  error: "오류",
  unknown: "미확인",
};

const CHARACTER_ROOM_LABEL: Record<OfficeMapNode["id"], string> = {
  sessions: "세션",
  work: "작업",
  automation: "자동화",
  routing: "라우팅",
};

const CHARACTER_ROLE_ROOM: Record<OfficeCharacterRole, OfficeMapNode["id"]> = {
  model: "sessions",
  operator: "sessions",
  worker: "work",
  reviewer: "work",
  automation_keeper: "automation",
  router: "routing",
  sentinel: "routing",
  alert: "work",
};

const CHARACTER_ROLE_KIND: Record<OfficeCharacterRole, OfficeSceneObject["kind"]> = {
  model: "avatar",
  operator: "avatar",
  worker: "desk",
  reviewer: "desk",
  automation_keeper: "machine",
  router: "mail",
  sentinel: "machine",
  alert: "alert",
};

function officeCharacterHealth(status: OfficeCharacterStatus): OfficeMapNode["health"] {
  if (status === "error" || status === "blocked") return "error";
  if (status === "warning" || status === "unknown") return "partial";
  return "ok";
}

function characterStatusFromText(value: unknown): OfficeCharacterStatus {
  const text = typeof value === "string" ? value.toLowerCase() : "";
  if (text.includes("block")) return "blocked";
  if (text.includes("error") || text.includes("fail")) return "error";
  if (text.includes("idle") || text.includes("done")) return "idle";
  if (text.includes("review")) return "reviewing";
  if (text.includes("sched")) return "scheduled";
  if (text.includes("active") || text.includes("run") || text.includes("open") || text.includes("progress")) return "active";
  return "unknown";
}

function characterDetail(room: OfficeMapNode | undefined, status: OfficeCharacterStatus): string {
  const roomLabel = room?.label ?? "미확인 방";
  return `${roomLabel} 역할 투영 · 상태 ${status}`;
}

function addCharacters(
  characters: OfficeCharacter[],
  role: OfficeCharacterRole,
  count: number,
  statusForIndex: (index: number) => OfficeCharacterStatus,
  nodesById: Map<OfficeMapNode["id"], OfficeMapNode>,
) {
  if (count <= 0) return;
  const roomId = CHARACTER_ROLE_ROOM[role];
  const room = nodesById.get(roomId);
  const slots = SCENE_SLOTS[roomId];
  const visible = Math.min(count, CHARACTER_LIMIT_PER_ROLE);
  for (let index = 0; index < visible; index += 1) {
    const [x, y] = slots[index];
    const status = statusForIndex(index);
    characters.push({
      id: `${role}-${index + 1}`,
      role,
      roomId,
      label: `${CHARACTER_ROLE_LABEL[role]} ${index + 1}`,
      status,
      detail: characterDetail(room, status),
      redactionNote: "안전 DTO 역할/상태/개수만 반영 · 원문 제외",
      x,
      y,
    });
  }
  if (count > CHARACTER_LIMIT_PER_ROLE) {
    const status: OfficeCharacterStatus = "warning";
    characters.push({
      id: `${role}-overflow`,
      role,
      roomId,
      label: `+${count - CHARACTER_LIMIT_PER_ROLE} ${CHARACTER_ROLE_LABEL[role]}`,
      status,
      detail: `${room?.label ?? "방"} 밀도 초과 역할 개수`,
      redactionNote: "지도 밀도 때문에 합산한 안전 개수 · 원문 제외",
      x: Math.min((room?.x ?? 78) + 12, 90),
      y: Math.min((room?.y ?? 68) + 11, 88),
    });
  }
}

export function buildOfficeCharacterView(character: OfficeCharacter): OfficeCharacterView {
  return {
    glyph: CHARACTER_ROLE_GLYPH[character.role],
    bodyClassName: `office-character office-character--${character.role} office-character--${character.status}`,
    accessoryClassName: `office-character__accessory office-character__accessory--${character.role}`,
    nameplate: CHARACTER_ROLE_NAMEPLATE[character.role],
    statusLabel: CHARACTER_STATUS_LABEL[character.status],
    safeTitle: `${CHARACTER_ROOM_LABEL[character.roomId]} · ${CHARACTER_ROLE_LABEL[character.role]} · ${CHARACTER_STATUS_LABEL[character.status]} · 안전 DTO 기반 · 원문 제외`,
  };
}

function characterRoomHasDelta(character: OfficeCharacter, delta: OfficeStateDelta): boolean {
  return delta.nodeBadges[character.roomId]?.length > 0 || delta.changedFlows.some((flow) => flow.from === character.roomId || flow.to === character.roomId);
}

function characterActivityId(character: OfficeCharacter, delta: OfficeStateDelta): OfficeCharacterActivityId {
  if (character.status === "blocked" || character.status === "error" || character.role === "alert") return "blocked";
  if (character.status === "warning") return "warning";
  if (character.status === "unknown") return "unknown";
  if (character.status === "idle") return "idle";
  if (character.role === "automation_keeper" && characterRoomHasDelta(character, delta)) return "soon";
  if (character.status === "scheduled" || character.role === "automation_keeper") return "scheduled";
  if (character.status === "reviewing" || character.role === "reviewer") return "reviewing";
  if (character.status === "routing" || character.role === "router") return "routing";
  if (character.status === "working" || character.role === "worker") return "working";
  if (character.status === "active" && (character.role === "model" || character.role === "operator")) return "thinking";
  return "idle";
}

export function buildOfficeCharacterActivity(character: OfficeCharacter, delta: OfficeStateDelta): OfficeCharacterActivity {
  const id = characterActivityId(character, delta);
  const style = CHARACTER_ACTIVITY_STYLE[id];
  const label = CHARACTER_ACTIVITY_LABEL[id];
  return {
    id,
    label,
    motion: style.motion,
    tone: style.tone,
    reducedMotionLabel: `${label} · 안전 상태/변화만 반영`,
  };
}

function routeTone(tone: OfficeFlowChange["tone"]): OfficeCharacterRoute["tone"] {
  if (tone === "negative") return "danger";
  if (tone === "warning") return "warning";
  return "normal";
}

function routeDetail(from: OfficeMapNode["id"], to: OfficeMapNode["id"]): string {
  if (from === "sessions" && to === "work") return "세션에서 작업으로 · 방금 변경";
  if (from === "work" && to === "automation") return "작업에서 자동화로 · 방금 변경";
  if (from === "automation" && to === "routing") return "자동화에서 라우팅으로 · 방금 변경";
  return `${CHARACTER_ROOM_LABEL[from]}에서 ${CHARACTER_ROOM_LABEL[to]}로 · 방금 변경`;
}

export function buildOfficeCharacterRoutes(delta: OfficeStateDelta): OfficeCharacterRoute[] {
  return delta.changedFlows.map((flow) => {
    const detail = routeDetail(flow.from, flow.to);
    const tone = routeTone(flow.tone);
    return {
      id: `route:${flow.from}->${flow.to}`,
      fromRoomId: flow.from,
      toRoomId: flow.to,
      label: "흐름 변경",
      detail,
      tone,
      motion: tone === "normal" ? "route" : "alert",
      changed: true,
      reducedMotionLabel: `${detail} · 애니메이션 없이 표시`,
    };
  });
}

function characterRecentChangeLabel(character: OfficeCharacter, delta: OfficeStateDelta): string {
  const badges = delta.nodeBadges[character.roomId] ?? [];
  const badgeText = badges.map((badge) => badge.label);
  const flowText = delta.changedFlows
    .filter((flow) => flow.from === character.roomId || flow.to === character.roomId)
    .map((flow) => routeDetail(flow.from, flow.to));
  const changes = [...badgeText, ...flowText];
  return changes.length > 0 ? changes.join(" · ") : "최근 안전 변화 없음";
}

export function buildOfficeCharacterInspector(character: OfficeCharacter, delta: OfficeStateDelta): OfficeCharacterInspector {
  const view = buildOfficeCharacterView(character);
  const activity = buildOfficeCharacterActivity(character, delta);
  const roleLabel = CHARACTER_ROLE_LABEL[character.role];
  const roomLabel = CHARACTER_ROOM_LABEL[character.roomId];
  const statusLabel = CHARACTER_STATUS_LABEL[character.status];
  return {
    kind: "RPG 캐릭터",
    title: `${roomLabel} · ${roleLabel}`,
    ariaLabel: `${roleLabel} 살펴보기, 방 ${roomLabel}, 상태 ${statusLabel}, 액션 ${activity.label}`,
    fields: [
      ["캐릭터", roleLabel],
      ["역할", view.nameplate],
      ["방", roomLabel],
      ["상태", statusLabel],
      ["액션", activity.label],
      ["최근 안전 변화", characterRecentChangeLabel(character, delta)],
      ["가림", "안전 DTO 역할/상태/개수/흐름만 반영 · 원문 제외"],
    ],
  };
}

export function buildOfficeCharacters(state: OfficeState, nodes: OfficeMapNode[]): OfficeCharacter[] {
  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const characters: OfficeCharacter[] = [];
  const blockedWorkCount = state.work_items.filter((item) => characterStatusFromText(item.status) === "blocked").length;
  const failedAutomationCount = state.automations.filter((job) => characterStatusFromText(job.state) === "error" || characterStatusFromText(job.last_status) === "error").length;
  const sourceWarnings = state.data_sources.filter((source) => source.status !== "ok" || (source.warning_count ?? 0) > 0);
  const routingCount = state.topics.length + state.provenance.length;

  addCharacters(characters, "model", state.agents.length, (index) => characterStatusFromText(state.agents[index]?.status), nodesById);
  addCharacters(characters, "worker", state.work_items.length, (index) => {
    const status = characterStatusFromText(state.work_items[index]?.status);
    return status === "blocked" ? "blocked" : status === "unknown" ? "working" : "working";
  }, nodesById);
  addCharacters(characters, "automation_keeper", state.automations.length, (index) => {
    const job = state.automations[index];
    const status = characterStatusFromText(job?.state ?? job?.last_status);
    if (status === "error") return "error";
    if (status === "unknown") return "scheduled";
    return status === "active" ? "scheduled" : status;
  }, nodesById);
  addCharacters(characters, "router", Math.max(routingCount, state.data_sources.some((source) => source.id === "topics" && source.status !== "ok") ? 1 : 0), () => (routingCount > 0 ? "routing" : "unknown"), nodesById);
  addCharacters(characters, "sentinel", sourceWarnings.length, (index) => {
    const status = sourceWarnings[index]?.status;
    if (status === "error") return "error";
    if (status === "partial" || status === "unavailable" || status === "missing") return "warning";
    return "unknown";
  }, nodesById);
  addCharacters(characters, "alert", blockedWorkCount + failedAutomationCount, () => "blocked", nodesById);

  return characters;
}

export function buildOfficeCharacterSceneObjects(characters: OfficeCharacter[]): OfficeSceneObject[] {
  return characters.map((character) => ({
    id: `character:${character.id}`,
    roomId: character.roomId,
    kind: CHARACTER_ROLE_KIND[character.role],
    label: character.label,
    detail: `${character.detail} · ${character.redactionNote}`,
    health: officeCharacterHealth(character.status),
    x: character.x,
    y: character.y,
  }));
}

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

export function buildOfficeSceneMotionTrack(object: OfficeSceneObject): OfficeSceneMotionTrack {
  const indexMatch = object.id.match(/-(\d+)$/);
  const index = indexMatch ? Number(indexMatch[1]) : 1;
  const direction = index % 2 === 0 ? -1 : 1;
  const delay = `-${(index * 0.3).toFixed(1)}s`;
  const baseStyle = {
    "--office-motion-delay": delay,
  };

  if (object.kind === "machine") {
    return {
      className: "office-scene-marker-motion office-scene-marker-blink",
      style: {
        ...baseStyle,
        "--office-motion-x": "0px",
        "--office-motion-y": "0px",
        "--office-motion-duration": `${(3.6 + (index % 3) * 0.4).toFixed(1)}s`,
      },
      ariaLabel: `${object.label} 상태등 표시 · 안전 DTO 기반`,
    };
  }

  if (object.kind === "desk") {
    return {
      className: "office-scene-marker-motion office-scene-marker-idle",
      style: {
        ...baseStyle,
        "--office-motion-x": "0px",
        "--office-motion-y": `${direction}px`,
        "--office-motion-duration": `${(5.8 + (index % 2) * 0.5).toFixed(1)}s`,
      },
      ariaLabel: `${object.label} 대기 표시 · 안전 DTO 기반`,
    };
  }

  const isAlert = object.kind === "alert";
  return {
    className: `office-scene-marker-motion ${isAlert ? "office-scene-marker-blink" : "office-scene-marker-walk"}`,
    style: {
      ...baseStyle,
      "--office-motion-x": `${isAlert ? 1 : 3 * direction}px`,
      "--office-motion-y": `${isAlert ? -1 : -2 * direction}px`,
      "--office-motion-duration": `${(4.5 + (index % 4) * 0.3).toFixed(1)}s`,
    },
    ariaLabel: `${object.label} ${isAlert ? "상태등" : "이동"} 표시 · 안전 DTO 기반`,
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
