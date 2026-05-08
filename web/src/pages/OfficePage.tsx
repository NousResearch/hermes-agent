import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  Building2,
  Clock,
  Database,
  Eye,
  Filter,
  Lock,
  MapPinned,
  RefreshCw,
  Route,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type OfficeDataSource, type OfficeSourceStatus, type OfficeState } from "@/lib/api";
import {
  buildOfficeAttentionItems,
  buildOfficeMapFlows,
  buildOfficeMapNodes,
  buildOfficeSceneObjectView,
  buildOfficeSceneObjects,
  groupByText,
  numberField,
  textField,
  visibleRows,
  type OfficeMapFlow,
  type OfficeMapNode,
  type OfficeSceneObject,
} from "./officeView";

const FOCUS_OPTIONS = ["overview", "work", "automation", "routing"] as const;
const LIST_LIMIT = 6;
const EVENT_LIMIT = 12;
type FocusOption = (typeof FOCUS_OPTIONS)[number];

const FOCUS_LABEL: Record<FocusOption, string> = {
  overview: "전체",
  work: "작업",
  automation: "자동화",
  routing: "라우팅",
};

const HEALTH_LABEL: Record<OfficeMapNode["health"], string> = {
  ok: "정상",
  partial: "부분 연결",
  missing: "미연결",
  error: "오류",
};

const ZONE_LABEL: Record<OfficeMapNode["zone"], string> = {
  entry: "입구",
  workbench: "작업대",
  machine: "기계실",
  routing: "라우팅",
};

type InspectorSelection = {
  kind: string;
  title: string;
  fields: Array<[string, string]>;
};

function fmt(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return new Date(value * 1000).toLocaleString();
  if (typeof value !== "string") return String(value);
  if (/^\d{4}-\d{2}-\d{2}T/.test(value)) return new Date(value).toLocaleString();
  return value;
}

const SOURCE_TONE: Record<OfficeSourceStatus, string> = {
  ok: "border-emerald-400/40 text-emerald-300",
  partial: "border-yellow-400/40 text-yellow-300",
  missing: "border-sky-400/40 text-sky-300",
  unavailable: "border-zinc-400/40 text-zinc-300",
  error: "border-red-400/40 text-red-300",
};

const SOURCE_LABEL: Record<OfficeSourceStatus, string> = {
  ok: "정상",
  partial: "부분 연결",
  missing: "미연결",
  unavailable: "사용 불가",
  error: "오류",
};

function StatusPill({ status }: { status: OfficeSourceStatus | string }) {
  const tone = SOURCE_TONE[status as OfficeSourceStatus] ?? "border-zinc-400/40 text-zinc-300";
  const label = SOURCE_LABEL[status as OfficeSourceStatus] ?? status;
  return (
    <span className={`whitespace-nowrap border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] ${tone}`}>
      {label}
    </span>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-midground/60">{children}</div>;
}

function SourceCard({ source, onInspect }: { source: OfficeDataSource; onInspect: () => void }) {
  return (
    <div className="border border-current/15 bg-black/20 p-3">
      <div className="flex items-center justify-between gap-3">
        <span className="text-sm font-semibold text-foreground">{source.id}</span>
        <StatusPill status={source.status} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-midground/75">
        <div>
          <div className="text-midground/45">항목</div>
          <div className="text-foreground">{source.item_count ?? "—"}</div>
        </div>
        <div>
          <div className="text-midground/45">경고</div>
          <div className={source.warning_count ? "text-yellow-300" : "text-foreground"}>{source.warning_count ?? 0}</div>
        </div>
      </div>
      {source.error_summary ? (
        <div className="mt-3 border border-red-400/30 bg-red-950/20 p-2 text-xs text-red-300/90">{source.error_summary}</div>
      ) : null}
      <button type="button" onClick={onInspect} className="mt-3 flex items-center gap-1 text-xs uppercase tracking-[0.16em] text-midground/70 hover:text-foreground">
        <Eye className="h-3 w-3" /> 살펴보기
      </button>
    </div>
  );
}

function EmptyLine({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="border border-dashed border-current/15 bg-black/10 p-4 text-sm text-midground/65">
      <div>가려진 OfficeState DTO에 {label} 정보가 없습니다.</div>
      {hint ? <div className="mt-2 text-xs leading-5 text-midground/50">{hint}</div> : null}
    </div>
  );
}

function StatCard({ label, value, detail, tone = "text-foreground" }: { label: string; value: unknown; detail: string; tone?: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-xs uppercase tracking-[0.18em] text-midground/70">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className={`text-3xl font-semibold ${tone}`}>{String(value ?? 0)}</div>
        <div className="mt-2 text-xs text-midground/60">{detail}</div>
      </CardContent>
    </Card>
  );
}

function EntityRow({
  title,
  meta,
  badge,
  warning,
  onInspect,
}: {
  title: string;
  meta: string;
  badge?: string;
  warning?: string | null;
  onInspect?: () => void;
}) {
  return (
    <div className="border border-current/15 bg-black/15 p-3 text-sm">
      <div className="flex items-start justify-between gap-3">
        <span className="font-semibold text-foreground">{title}</span>
        {badge ? <span className="shrink-0 text-xs text-midground/70">{badge}</span> : null}
      </div>
      <div className="mt-1 text-xs text-midground/70">{meta}</div>
      {warning ? <div className="mt-2 border border-red-400/30 bg-red-950/20 p-2 text-xs text-red-300">{warning}</div> : null}
      {onInspect ? (
        <button type="button" onClick={onInspect} className="mt-3 flex items-center gap-1 text-xs uppercase tracking-[0.16em] text-midground/70 hover:text-foreground">
          <Eye className="h-3 w-3" /> 살펴보기
        </button>
      ) : null}
    </div>
  );
}

function MiniList({
  title,
  icon,
  children,
  meta,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  meta?: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          {icon}
          {title}
        </CardTitle>
        {meta ? <div className="text-xs text-midground/55">{meta}</div> : null}
      </CardHeader>
      <CardContent className="space-y-3">{children}</CardContent>
    </Card>
  );
}

function mapNodeTone(health: OfficeMapNode["health"]): string {
  if (health === "ok") return "border-emerald-300/70 bg-emerald-950/70 text-emerald-50";
  if (health === "partial") return "border-yellow-300/75 bg-yellow-950/70 text-yellow-50";
  if (health === "error") return "border-red-300/80 bg-red-950/70 text-red-50";
  return "border-sky-300/65 bg-sky-950/70 text-sky-50";
}

function mapFlowTone(health: OfficeMapFlow["health"]): string {
  if (health === "ok") return "text-emerald-200/45";
  if (health === "partial") return "text-yellow-200/55";
  if (health === "error") return "text-red-200/60";
  return "text-sky-200/45";
}

function SceneObjectMarker({ object }: { object: OfficeSceneObject }) {
  const view = buildOfficeSceneObjectView(object);
  return (
    <div
      className={`pointer-events-none absolute z-20 flex h-5 w-5 -translate-x-1/2 -translate-y-1/2 items-center justify-center border text-[10px] font-bold shadow-md ring-1 ring-black/50 ${view.toneClass}`}
      style={{ left: `${object.x}%`, top: `${object.y}%` }}
      title={view.title}
      aria-hidden={view.ariaHidden}
      data-office-scene-marker="true"
    >
      {view.glyph}
    </div>
  );
}

const OFFICE_ZONE_PANELS: Array<{ id: OfficeMapNode["id"]; label: string; className: string; style: React.CSSProperties }> = [
  { id: "sessions", label: "입구", className: "border-emerald-200/25 bg-[repeating-linear-gradient(45deg,rgba(16,185,129,0.13)_0_6px,rgba(16,185,129,0.055)_6px_12px)]", style: { left: "10%", top: "14%", width: "34%", height: "30%" } },
  { id: "work", label: "작업대", className: "border-yellow-200/25 bg-[repeating-linear-gradient(0deg,rgba(234,179,8,0.13)_0_5px,rgba(234,179,8,0.055)_5px_11px)]", style: { left: "56%", top: "14%", width: "34%", height: "30%" } },
  { id: "automation", label: "기계실", className: "border-cyan-200/25 bg-[repeating-linear-gradient(90deg,rgba(34,211,238,0.13)_0_5px,rgba(34,211,238,0.055)_5px_11px)]", style: { left: "10%", top: "54%", width: "34%", height: "25%" } },
  { id: "routing", label: "우편실", className: "border-sky-200/25 bg-[repeating-linear-gradient(135deg,rgba(125,211,252,0.13)_0_6px,rgba(125,211,252,0.055)_6px_12px)]", style: { left: "56%", top: "54%", width: "34%", height: "25%" } },
];

function OfficeMap({
  nodes,
  flows,
  sceneObjects,
  onInspect,
}: {
  nodes: OfficeMapNode[];
  flows: OfficeMapFlow[];
  sceneObjects: OfficeSceneObject[];
  onInspect: (node: OfficeMapNode) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <MapPinned className="h-4 w-4" /> 오피스 맵
        </CardTitle>
        <div className="text-xs text-midground/55">
          가려진 DTO 개수만으로 브라우저 안에서 그리는 CSS/SVG 평면도입니다. 픽셀 엔진, 새 의존성, 변경 제어는 없습니다.
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative min-h-[560px] overflow-hidden border border-current/20 bg-[radial-gradient(circle_at_top_left,rgba(16,185,129,0.16),transparent_34%),linear-gradient(135deg,rgba(255,255,255,0.055),rgba(0,0,0,0.20))] p-4 sm:min-h-[510px]">
          <svg className="pointer-events-none absolute inset-0 z-10 h-full w-full text-midground/20" role="img" aria-label="읽기 전용 오피스 흐름 연결" viewBox="0 0 100 100" preserveAspectRatio="none">
            <defs>
              <marker id="office-map-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M0 0 L8 4 L0 8 Z" fill="currentColor" />
              </marker>
            </defs>
            <rect x="8" y="12" width="84" height="76" fill="none" stroke="currentColor" strokeWidth="0.5" />
            <path d="M50 12 V88" fill="none" stroke="currentColor" strokeWidth="0.35" opacity="0.5" />
            {flows.map((flow) => {
              const from = nodes.find((node) => node.id === flow.from);
              const to = nodes.find((node) => node.id === flow.to);
              if (!from || !to) return null;
              return (
                <path
                  key={`${flow.from}-${flow.to}`}
                  d={`M${from.x} ${from.y} L${to.x} ${to.y}`}
                  fill="none"
                  stroke="currentColor"
                  strokeDasharray={flow.health === "ok" ? "" : "2 2"}
                  strokeWidth="0.55"
                  markerEnd="url(#office-map-arrow)"
                  className={mapFlowTone(flow.health)}
                />
              );
            })}
          </svg>
          <div className="absolute left-4 top-4 z-40 border border-current/10 bg-black/35 px-2 py-1 text-[10px] uppercase tracking-[0.22em] text-midground/80">안전 오피스 투영</div>
          {OFFICE_ZONE_PANELS.map((zone) => (
            <div key={zone.id} className={`absolute z-0 border shadow-inner ${zone.className}`} style={zone.style} aria-hidden="true">
              <div className="absolute bottom-2 right-2 border border-current/10 bg-black/35 px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-[0.18em] text-midground/70">{zone.label}</div>
            </div>
          ))}
          {sceneObjects.map((object) => (
            <SceneObjectMarker key={object.id} object={object} />
          ))}
          {nodes.map((node) => (
            <button
              key={node.id}
              type="button"
              onClick={() => onInspect(node)}
              aria-label={`${node.label} 오피스 맵 방, 안전 항목 ${node.count}개, 상태 ${HEALTH_LABEL[node.health]}`}
              className={`absolute z-30 w-[min(9.25rem,42vw)] -translate-x-1/2 -translate-y-1/2 border p-2 text-left shadow-xl ring-1 ring-black/40 backdrop-blur-md transition hover:scale-[1.02] hover:border-current/70 focus:outline-none focus:ring-2 focus:ring-emerald-200/70 ${mapNodeTone(node.health)}`}
              style={{ left: `${node.x}%`, top: `${node.y}%` }}
            >
              <div className="text-[9px] font-semibold uppercase tracking-[0.18em] text-current/70">{ZONE_LABEL[node.zone]}</div>
              <div className="mt-1 flex items-center justify-between gap-3">
                <span className="text-[13px] font-bold uppercase tracking-[0.14em]">{node.label}</span>
                <span className="text-2xl font-bold">{node.count}</span>
              </div>
              <div className="mt-2 text-[11px] leading-4 text-current/85">{node.detail}</div>
              <div className="mt-3 text-[10px] font-semibold uppercase tracking-[0.16em] text-current/75">{HEALTH_LABEL[node.health]}</div>
            </button>
          ))}
          <div className="absolute bottom-4 left-4 right-4 z-40 border border-current/15 bg-black/50 p-3 text-xs leading-5 text-midground/80 shadow-lg backdrop-blur-sm">
            <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] uppercase tracking-[0.16em]">
              {flows.map((flow) => (
                <span key={`${flow.from}-${flow.to}`} className={mapFlowTone(flow.health)}>{flow.label} · {HEALTH_LABEL[flow.health]}</span>
              ))}
            </div>
            이 지도는 시각 인덱스입니다. 원문 프롬프트, 대화 기록, cron 스크립트, 작업 본문, 로그, 인증 정보, 비밀값은 브라우저 DTO 밖에 둡니다.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function GroupBlock({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <div className="border border-current/10 bg-black/10 p-3">
      <div className="mb-3 flex items-center justify-between gap-3 text-xs uppercase tracking-[0.16em] text-midground/65">
        <span>{title}</span>
        <span>{count}</span>
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function LimitedRows<T>({
  rows,
  limit = LIST_LIMIT,
  label,
  children,
}: {
  rows: T[];
  limit?: number;
  label: string;
  children: (row: T) => React.ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const visible = visibleRows(rows, limit, expanded);
  const hidden = Math.max(rows.length - visible.length, 0);
  return (
    <>
      {visible.map((row) => children(row))}
      {rows.length > limit ? (
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="w-full border border-dashed border-current/20 bg-black/10 px-3 py-2 text-left text-xs uppercase tracking-[0.16em] text-midground/65 hover:text-foreground"
        >
          {expanded ? `${label} 접기` : `${label} ${hidden}개 더 보기`}
        </button>
      ) : null}
    </>
  );
}

function InspectorPanel({ selection }: { selection: InspectorSelection | null }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Eye className="h-4 w-4" /> 안전 정보 살펴보기
        </CardTitle>
      </CardHeader>
      <CardContent>
        {selection ? (
          <div className="space-y-3 text-sm">
            <div>
              <SectionLabel>{selection.kind}</SectionLabel>
              <div className="mt-1 font-semibold text-foreground">{selection.title}</div>
            </div>
            <div className="grid gap-2">
              {selection.fields.map(([label, value]) => (
                <div key={label} className="grid grid-cols-[8rem_1fr] gap-3 border border-current/10 bg-black/15 p-2 text-xs">
                  <span className="text-midground/50">{label}</span>
                  <span className="break-words text-midground/85">{value}</span>
                </div>
              ))}
            </div>
            <div className="border border-emerald-400/20 bg-emerald-950/10 p-3 text-xs leading-5 text-emerald-200/80">
              이 패널은 DTO 메타데이터만 보여줍니다. 원문 프롬프트, 대화 기록, 작업 본문, cron 스크립트, 로그, 인증 정보, 비밀값은 계속 제외됩니다.
            </div>
          </div>
        ) : (
          <div className="border border-dashed border-current/15 bg-black/10 p-4 text-sm text-midground/65">
            소스, 방, 세션, 작업, 자동화, 토픽, 이벤트에서 살펴보기를 누르면 안전 메타데이터가 여기에 표시됩니다.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function OfficePage() {
  const [state, setState] = useState<OfficeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [focus, setFocus] = useState<FocusOption>("overview");
  const [selection, setSelection] = useState<InspectorSelection | null>(null);

  const load = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const next = await api.getOfficeState();
      setState(next);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .getOfficeState()
      .then((next) => {
        if (!cancelled) setState(next);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const inspectRecord = useCallback((kind: string, title: string, fields: Array<[string, string]>) => {
    setSelection({ kind, title, fields });
  }, []);

  const needsAttention = useMemo(() => (state ? buildOfficeAttentionItems(state) : []), [state]);
  const mapNodes = useMemo(() => (state ? buildOfficeMapNodes(state) : []), [state]);
  const mapFlows = useMemo(() => buildOfficeMapFlows(mapNodes), [mapNodes]);
  const sceneObjects = useMemo(() => (state ? buildOfficeSceneObjects(state, mapNodes) : []), [state, mapNodes]);

  const sourceCounts = useMemo(() => {
    if (!state) return { ok: 0, partial: 0, missing: 0, unavailable: 0, error: 0 };
    return state.data_sources.reduce<Record<OfficeSourceStatus, number>>(
      (acc, source) => {
        acc[source.status] += 1;
        return acc;
      },
      { ok: 0, partial: 0, missing: 0, unavailable: 0, error: 0 },
    );
  }, [state]);

  const workGroups = useMemo(() => (state ? groupByText(state.work_items, "status", "unknown") : {}), [state]);
  const automationGroups = useMemo(() => (state ? groupByText(state.automations, "state", "unknown") : {}), [state]);

  const showOverview = focus === "overview";
  const showWork = focus === "overview" || focus === "work";
  const showAutomation = focus === "overview" || focus === "automation";
  const showRouting = focus === "overview" || focus === "routing";

  if (loading) {
    return (
      <div className="flex min-h-[420px] flex-col items-center justify-center gap-4 border border-current/15 bg-black/10 py-24">
        <Spinner className="text-2xl text-primary" />
        <div className="text-sm uppercase tracking-[0.2em] text-midground/70">가려진 OfficeState를 불러오는 중</div>
      </div>
    );
  }

  if (error || !state) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base text-red-300">
            <AlertTriangle className="h-4 w-4" /> 오피스를 사용할 수 없음
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-midground/80">
          <p>{error ?? "상태가 반환되지 않았습니다"}</p>
          <p className="text-xs text-midground/55">보호된 OfficeState DTO를 읽지 못했습니다. 이 대체 화면도 원문 로그나 비밀값은 노출하지 않습니다.</p>
          <Button onClick={load} className="gap-2 uppercase">
            <RefreshCw className="h-4 w-4" /> 다시 시도
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-6 normal-case">
      <div className="border border-current/20 bg-gradient-to-br from-black/35 to-black/10 p-5">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div className="max-w-3xl">
            <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.22em] text-emerald-300">
              <ShieldCheck className="h-4 w-4" /> 읽기 전용 MVP · 로컬호스트 우선
            </div>
            <h1 className="mt-3 text-3xl font-semibold uppercase tracking-wide text-foreground md:text-4xl">Hermes AI 오피스</h1>
            <p className="mt-3 text-sm leading-6 text-midground/80">
              이 Mac에서 도는 Hermes 상태를 가려서 보여주는 운영 지도입니다. 원문 세션, 프롬프트, 로그, 비밀값을 노출하지 않고 상태·건강도·출처 공백만 보여줍니다.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {FOCUS_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setFocus(option)}
                  className={`border px-3 py-1 text-xs uppercase tracking-[0.16em] ${focus === option ? "border-emerald-400/50 text-emerald-300" : "border-current/20 text-midground/70 hover:text-foreground"}`}
                >
                  {FOCUS_LABEL[option]}
                </button>
              ))}
            </div>
          </div>
          <div className="min-w-64 border border-current/15 bg-black/20 p-3 text-xs text-midground/70">
            <div className="flex items-center gap-2 text-foreground">
              <Lock className="h-4 w-4 text-emerald-300" /> 안전 모드
            </div>
            <div className="mt-2 grid gap-1">
              <div>생성 시각: {fmt(state.generated_at)}</div>
              <div>표시 모드: {state.display_mode}</div>
              <div>원격 모드: {state.capabilities.remote_mode}</div>
              <div>변경 기능: {state.capabilities.mutations_enabled ? "켜짐" : "없음"}</div>
            </div>
            <Button onClick={load} className="mt-4 w-full gap-2 uppercase" disabled={refreshing}>
              <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} /> 새로고침
            </Button>
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        <StatCard label="진행 중 작업" value={state.summary.active_work_count ?? 0} detail="승인된 어댑터가 보여준 열린 작업" />
        <StatCard label="확인 필요" value={needsAttention.length} detail="막힌 작업, 소스 경고, 실패한 자동화" tone={needsAttention.length > 0 ? "text-yellow-300" : "text-emerald-300"} />
        <StatCard label="자동화" value={state.summary.automation_count ?? state.automations.length} detail="읽기 전용 기계처럼 표시한 cron 작업" />
        <StatCard label="가림 처리" value={state.redactions.redacted_field_count} detail={`정책 v${state.redactions.policy_version}; 민감 원문 필드 제외`} />
      </div>

      {showOverview ? (
        <OfficeMap
          nodes={mapNodes}
          flows={mapFlows}
          sceneObjects={sceneObjects}
          onInspect={(node) => inspectRecord("오피스 맵 방", node.label, [
            ["방", node.id],
            ["구역", node.zone],
            ["안전 개수", String(node.count)],
            ["상태", node.health],
            ["설명", node.detail],
          ])}
        />
      ) : null}

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="h-4 w-4" /> 소스 상태
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="mb-4 flex flex-wrap gap-2 text-xs">
              <span className="border border-emerald-400/30 px-2 py-1 text-emerald-300">정상 {sourceCounts.ok}</span>
              <span className="border border-yellow-400/30 px-2 py-1 text-yellow-300">부분 연결 {sourceCounts.partial}</span>
              <span className="border border-sky-400/30 px-2 py-1 text-sky-300">미연결 {sourceCounts.missing}</span>
              <span className="border border-red-400/30 px-2 py-1 text-red-300">오류 {sourceCounts.error}</span>
            </div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {state.data_sources.map((source) => (
                <SourceCard
                  key={source.id}
                  source={source}
                  onInspect={() => inspectRecord("데이터 소스", source.id, [
                    ["상태", SOURCE_LABEL[source.status]],
                    ["확인 시각", fmt(source.checked_at)],
                    ["항목", String(source.item_count ?? "—")],
                    ["경고", String(source.warning_count ?? 0)],
                    ["오류", source.error_summary ?? "—"],
                  ])}
                />
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <AlertTriangle className="h-4 w-4" /> 확인 필요 목록
            </CardTitle>
          </CardHeader>
          <CardContent>
            {needsAttention.length === 0 ? (
              <div className="border border-emerald-400/25 bg-emerald-950/10 p-4 text-sm text-emerald-300">
                가려진 DTO에 막힌 작업, 실패한 자동화, 소스 경고가 없습니다.
              </div>
            ) : (
              <div className="space-y-2">
                <LimitedRows rows={needsAttention} limit={LIST_LIMIT} label="확인 필요 항목">
                  {(item) => (
                    <div key={item.id} className="border border-yellow-300/30 bg-yellow-950/10 p-3 text-sm text-yellow-200">
                      <span className="font-semibold">{item.label}</span>
                      <span className="ml-2 text-xs text-yellow-100/70">{item.detail}</span>
                    </div>
                  )}
                </LimitedRows>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_24rem]">
        <div className="flex flex-col gap-6">
          {showWork ? (
            <div className="grid gap-6 xl:grid-cols-2">
              <MiniList title="방 / 작업 흐름" icon={<Building2 className="h-4 w-4" />} meta="묶어서 보기 위한 화면일 뿐, 방이 원본 데이터는 아닙니다.">
                {state.rooms.length === 0 ? (
                  <EmptyLine label="방" hint="아직 칸반/토픽/시스템 방 투영이 없습니다. 방이 없다는 것이 작업이 비었다는 뜻은 아닙니다." />
                ) : (
                  <LimitedRows rows={state.rooms} label="방">
                    {(room) => (
                      <EntityRow
                        key={String(room.id)}
                        title={textField(room, "display_name")}
                        badge={textField(room, "kind")}
                        meta={`소스 ${textField(room, "source")} · ID ${String(room.id)}`}
                        onInspect={() => inspectRecord("방", textField(room, "display_name"), [
                          ["ID", String(room.id)],
                          ["종류", textField(room, "kind")],
                          ["소스", textField(room, "source")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                )}
              </MiniList>

              <MiniList title="세션 / 에이전트" icon={<Bot className="h-4 w-4" />} meta="세션 제목과 미리보기는 별도 허용 전까지 가립니다.">
                {state.agents.length === 0 ? (
                  <EmptyLine label="세션 메타데이터" hint="이번 스냅샷에서 세션 어댑터가 안전 메타데이터를 제공하지 않았습니다." />
                ) : (
                  <LimitedRows rows={state.agents} label="세션">
                    {(agent) => (
                      <EntityRow
                        key={String(agent.id)}
                        title={textField(agent, "source_platform")}
                        badge={textField(agent, "status")}
                        meta={`메시지 ${numberField(agent, "message_count") ?? 0}개 · 제목 ${textField(agent, "title_policy")}`}
                        onInspect={() => inspectRecord("세션 / 에이전트", textField(agent, "source_platform"), [
                          ["id", String(agent.id)],
                          ["상태", textField(agent, "status")],
                          ["메시지", String(numberField(agent, "message_count") ?? 0)],
                          ["제목 정책", textField(agent, "title_policy")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                )}
              </MiniList>
            </div>
          ) : null}

          {showWork ? (
            <MiniList title="작업 항목" icon={<MapPinned className="h-4 w-4" />} meta="안전 상태별로 묶어 보여줍니다. 본문/결과/댓글/로그는 제외합니다.">
              {state.work_items.length === 0 ? (
                <EmptyLine label="작업 항목" hint="승인된 어댑터가 작업 카드를 보고하지 않았습니다. 외부 보드가 모두 비었다는 뜻은 아닙니다." />
              ) : Object.entries(workGroups).map(([status, items]) => (
                <GroupBlock key={status} title={status} count={items.length}>
                  <LimitedRows rows={items} label="작업 항목">
                    {(item) => (
                      <EntityRow
                        key={String(item.id)}
                        title={textField(item, "title")}
                        badge={textField(item, "status")}
                        meta={`담당 ${textField(item, "assignee")} · 우선순위 ${numberField(item, "priority") ?? 0}`}
                        onInspect={() => inspectRecord("작업 항목", textField(item, "title"), [
                          ["id", String(item.id)],
                          ["상태", textField(item, "status")],
                          ["담당", textField(item, "assignee")],
                          ["우선순위", String(numberField(item, "priority") ?? 0)],
                        ])}
                      />
                    )}
                  </LimitedRows>
                </GroupBlock>
              ))}
            </MiniList>
          ) : null}

          {showAutomation ? (
            <MiniList title="자동화" icon={<Clock className="h-4 w-4" />} meta="작업 상태별로 묶어 보여줍니다. 실행/일시정지/재개/삭제 제어는 없습니다.">
              {state.automations.length === 0 ? (
                <EmptyLine label="자동화" hint="이번 스냅샷에서 읽기 전용 어댑터가 cron 스타일 작업을 제공하지 않았습니다." />
              ) : Object.entries(automationGroups).map(([jobState, jobs]) => (
                <GroupBlock key={jobState} title={jobState} count={jobs.length}>
                  <LimitedRows rows={jobs} label="자동화">
                    {(job) => (
                      <EntityRow
                        key={String(job.id)}
                        title={textField(job, "name")}
                        badge={textField(job, "state")}
                        meta={`최근 ${fmt(job.last_status)} · 다음 ${fmt(job.next_run_at)}`}
                        warning={typeof job.last_error_summary === "string" ? job.last_error_summary : null}
                        onInspect={() => inspectRecord("자동화", textField(job, "name"), [
                          ["id", String(job.id)],
                          ["상태", textField(job, "state")],
                          ["최근 상태", fmt(job.last_status)],
                          ["다음 실행", fmt(job.next_run_at)],
                          ["전달", textField(job, "delivery_policy")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                </GroupBlock>
              ))}
            </MiniList>
          ) : null}

          {showRouting ? (
            <div className="grid gap-6 xl:grid-cols-2">
              <MiniList title="토픽 라우팅" icon={<Route className="h-4 w-4" />} meta="읽기 전용 라우팅 투영입니다. 모르는 출처는 그대로 명시합니다.">
                {state.topics.length === 0 ? (
                  <EmptyLine label="토픽 라우팅 기록" hint="승인된 토픽 레지스트리/투영이 연결되어 있지 않습니다. UI 오류가 아니라 알려진 소스 공백입니다." />
                ) : (
                  <LimitedRows rows={state.topics} label="토픽">
                    {(topic) => (
                      <EntityRow
                        key={String(topic.id)}
                        title={textField(topic, "display_name")}
                        badge={textField(topic, "platform")}
                        meta={`목적 ${textField(topic, "purpose")} · 신뢰도 ${textField(topic, "confidence")}`}
                        onInspect={() => inspectRecord("토픽", textField(topic, "display_name"), [
                          ["id", String(topic.id)],
                          ["플랫폼", textField(topic, "platform")],
                          ["목적", textField(topic, "purpose")],
                          ["신뢰도", textField(topic, "confidence")],
                          ["source", textField(topic, "source")],
                        ])}
                      />
                    )}
                  </LimitedRows>
                )}
              </MiniList>

              <MiniList title="출처 / 가림 처리" icon={<Database className="h-4 w-4" />} meta="민감 원문이 아니라 개수와 정책만 보여줍니다.">
                <div className="grid gap-3 text-sm">
                  <div className="border border-current/15 bg-black/15 p-3">
                    <SectionLabel>출처 기록</SectionLabel>
                    <div className="mt-2 text-2xl text-foreground">{state.provenance.length}</div>
                    <div className="mt-1 text-xs text-midground/65">알 수 없거나 빠진 출처는 그대로 표시하며, 민감 텍스트에서 추론하지 않습니다.</div>
                  </div>
                  <div className="border border-current/15 bg-black/15 p-3">
                    <SectionLabel>제외된 섹션</SectionLabel>
                    <div className="mt-2 text-xs text-midground/75">
                      {state.redactions.omitted_sections.length === 0 ? "—" : state.redactions.omitted_sections.join(" · ")}
                    </div>
                  </div>
                  {state.redactions.warnings.length > 0 ? (
                    <div className="border border-yellow-300/30 bg-yellow-950/10 p-3 text-xs text-yellow-200">
                      {state.redactions.warnings.join(" · ")}
                    </div>
                  ) : null}
                  <button
                    type="button"
                    onClick={() => inspectRecord("가림 처리 보고서", `정책 v${state.redactions.policy_version}`, [
                      ["가린 필드", String(state.redactions.redacted_field_count)],
                      ["제외 섹션", state.redactions.omitted_sections.length === 0 ? "—" : state.redactions.omitted_sections.join(" · ")],
                      ["경고", state.redactions.warnings.length === 0 ? "—" : state.redactions.warnings.join(" · ")],
                    ])}
                    className="flex items-center gap-1 text-xs uppercase tracking-[0.16em] text-midground/70 hover:text-foreground"
                  >
                    <Eye className="h-3 w-3" /> 가림 처리 보고서 보기
                  </button>
                </div>
              </MiniList>
            </div>
          ) : null}

          {showOverview ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">최근 안전 이벤트</CardTitle>
              </CardHeader>
              <CardContent>
                {state.events.length === 0 ? (
                  <EmptyLine label="이벤트" hint="이번 스냅샷에서 안전 시간표가 생성되지 않았습니다. 원문 로그와 대화 기록은 설계상 숨깁니다." />
                ) : (
                  <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                    <LimitedRows rows={state.events} limit={EVENT_LIMIT} label="events">
                      {(event) => (
                        <button
                          type="button"
                          key={String(event.id)}
                          onClick={() => inspectRecord("이벤트", textField(event, "kind"), [
                            ["id", String(event.id)],
                            ["source", textField(event, "source")],
                            ["생성 시각", fmt(event.created_at)],
                          ])}
                          className="border border-current/15 bg-black/15 p-2 text-left text-xs hover:border-current/30"
                        >
                          <div className="font-semibold text-foreground">{textField(event, "kind")}</div>
                          <div className="mt-1 text-midground/70">{textField(event, "source")} · {fmt(event.created_at)}</div>
                        </button>
                      )}
                    </LimitedRows>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : null}
        </div>

        <div className="xl:sticky xl:top-4 xl:self-start">
          <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-midground/55">
            <Filter className="h-3 w-3" /> 보기: {FOCUS_LABEL[focus]}
          </div>
          <InspectorPanel selection={selection} />
        </div>
      </div>
    </div>
  );
}
