import { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  Brain,
  CheckCircle2,
  Circle,
  Clock,
  GitBranch,
  History,
  Layers,
  LineChart,
  MessageSquare,
  Network,
  RefreshCw,
  Send,
  TrendingDown,
  TrendingUp,
  X,
  Zap,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

// -------- Types --------
interface ScoreSignal {
  // Backend returns breakdown as {name: [emoji, reason]}
  // but Python tuples serialize as arrays in JSON
  [key: string]: [string, string];
}

interface DualAgentStatus {
  data: {
    ts: string;
    openclaw: {
      gateway_state?: string;
      last_exit_code?: string;
      codex_app_server_running?: boolean;
      error?: string;
    };
    sync_bridge: {
      overall_ok?: boolean;
      checks?: Array<{ id: string; ok: boolean; detail: string }>;
      error?: string;
    };
    bus: {
      open_count?: number;
      status_counts?: Record<string, number>;
      events_7d?: Record<string, number>;
      keep_alive_7d?: number;
      amend_learning_7d?: number;
      error?: string;
    };
    evolution: {
      total_outbound_hermes_to_openclaw?: number;
      total_inbound_openclaw_to_hermes?: number;
      this_week_outbound?: number;
      this_week_inbound?: number;
      m2_symmetry_ok?: boolean;
      inbound_files?: string[];
      outbound_files?: string[];
    };
    hf: {
      total?: number;
      by_status?: Record<string, number>;
      oldest_pending_hours?: number;
      pending_files?: string[];
      health_ok?: boolean;
    };
    queue: {
      total?: number;
      fresh?: number;
      stale?: number;
      force_review?: number;
      pilot_at_risk?: number;
      terminal?: number;
    };
    coaching: {
      blanks_openclaw?: number;
      empty_chair_filled?: number;
    };
    ratelimit?: {
      total_ever?: number;
      last_7d?: number;
      last_24h?: number;
    };
    daily_review?: {
      hermes_last_reviewed?: string;
      latest_change_log_date?: string;
      today_done?: boolean;
    };
  };
  scorecard: {
    score: number;
    max: number;
    breakdown: ScoreSignal;
    trend: string;
    next_actions: string[];
  };
}

interface BusTask {
  task_id: string;
  status: string;
  from_agent: string;
  to_agent: string;
  goal: string;
  result: string;
  created_at: number;
  acked_at: number | null;
  completed_at: number | null;
}

interface HFCard {
  file: string;
  title: string;
  status: string;
  from_agent: string;
  to_agent: string;
  freshness_ts: string;
  mtime: number;
}

interface ActivityEvent {
  kind: string;
  title: string;
  subtitle?: string;
  path?: string;
  status?: string;
  agent?: string;
  ts: number;
  source: string;
}

interface Snapshot {
  file: string;
  ts: string;
  score: number | null;
}

interface ThrottleState {
  enforcement_enabled: boolean;
  pairs: Record<string, { allowed: boolean; reason: string | null; stats: Record<string, unknown> }>;
}

interface MiddlewareEntry {
  name: string;
  order: number;
  env_var: string | null;
  critical: boolean;
  enabled: boolean;
}

interface MiddlewareStatus {
  master_enabled: boolean;
  master_mode: string;
  entries: MiddlewareEntry[];
}

interface AutoMemoryFact {
  id: string;
  content: string;
  category: string;
  confidence: number;
  created_at: number;
  source: string;
}

interface AutoMemoryData {
  exists: boolean;
  path: string;
  count?: number;
  updated_at?: number;
  facts: AutoMemoryFact[];
  error?: string;
}

// -------- Helpers --------
function statusBadgeVariant(status: string): "success" | "warning" | "destructive" | "outline" {
  if (status === "done") return "success";
  if (status === "fail" || status === "timeout") return "destructive";
  if (status === "ack" || status === "progress" || status === "keep-alive") return "warning";
  if (status === "pending") return "warning";
  return "outline";
}

function hfStatusBadge(status: string): "success" | "warning" | "destructive" | "outline" {
  if (status.startsWith("accepted")) return "success";
  if (status.includes("rejected")) return "destructive";
  if (status.includes("pending")) return "warning";
  return "outline";
}

function timeAgoShort(epoch: number): string {
  const sec = Date.now() / 1000 - epoch;
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.round(sec / 60)}m`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h`;
  return `${Math.round(sec / 86400)}d`;
}

// -------- Page --------
export default function AgentsPage() {
  const [status, setStatus] = useState<DualAgentStatus | null>(null);
  const [bus, setBus] = useState<BusTask[]>([]);
  const [hf, setHf] = useState<HFCard[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showDispatch, setShowDispatch] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [activity, setActivity] = useState<ActivityEvent[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [throttle, setThrottle] = useState<ThrottleState | null>(null);
  const [middleware, setMiddleware] = useState<MiddlewareStatus | null>(null);
  const [autoMemory, setAutoMemory] = useState<AutoMemoryData | null>(null);
  const [showCoaching, setShowCoaching] = useState(false);

  function flashToast(msg: string) {
    setToast(msg);
    setTimeout(() => setToast(null), 4000);
  }

  function refresh() {
    setRefreshNonce((n) => n + 1);
  }

  useEffect(() => {
    const load = async () => {
      try {
        const [s, b, h, a, sn, th, mw, am] = await Promise.all([
          fetch("/api/dual-agent/status").then((r) => r.json()),
          fetch("/api/dual-agent/bus").then((r) => r.json()),
          fetch("/api/dual-agent/handoffs").then((r) => r.json()),
          fetch("/api/dual-agent/activity?hours=24").then((r) => r.json()),
          fetch("/api/dual-agent/snapshots").then((r) => r.json()),
          fetch("/api/dual-agent/throttle").then((r) => r.json()),
          fetch("/api/dual-agent/middleware").then((r) => r.json()),
          fetch("/api/dual-agent/auto-memory").then((r) => r.json()),
        ]);
        if (s.error) setErr(s.error);
        else setStatus(s as DualAgentStatus);
        setBus((b.tasks || []) as BusTask[]);
        setHf((h.cards || []) as HFCard[]);
        setActivity((a.events || []) as ActivityEvent[]);
        setSnapshots((sn.snapshots || []) as Snapshot[]);
        setThrottle(th as ThrottleState);
        setMiddleware(mw as MiddlewareStatus);
        setAutoMemory(am as AutoMemoryData);
        setLoading(false);
      } catch (e) {
        setErr(String(e));
        setLoading(false);
      }
    };
    load();
    const iv = setInterval(load, 30000);
    return () => clearInterval(iv);
  }, [refreshNonce]);

  async function closeBusTask(taskId: string, outcome: "done" | "fail" | "keep-alive") {
    const summary = prompt(
      `為 ${taskId} 下 ${outcome} — 請輸入一句 summary:`,
      outcome === "keep-alive" ? "仍在進行中，延長 deadline" : ""
    );
    if (!summary || !summary.trim()) return;
    const res = await fetch(`/api/dual-agent/bus/${taskId}/close`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outcome, summary }),
    });
    const j = await res.json();
    if (j.ok) {
      flashToast(`✅ ${taskId} → ${outcome}`);
      refresh();
    } else {
      flashToast(`❌ ${j.error || "close failed"}`);
    }
  }

  async function hfDecision(
    cardFile: string,
    decision: "accept" | "reject" | "request-clarification"
  ) {
    const note = prompt(`為 ${cardFile} 加一句 decision note（可空）：`, "") ?? "";
    const res = await fetch(
      `/api/dual-agent/hf/${encodeURIComponent(cardFile)}/decision`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, note: note.trim() || null }),
      }
    );
    const j = await res.json();
    if (j.ok) {
      flashToast(`✅ ${cardFile} → ${j.new_status}`);
      refresh();
    } else {
      flashToast(`❌ ${j.error || "decision failed"}`);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (err) {
    return (
      <div className="mx-auto max-w-4xl p-6">
        <Card>
          <CardContent className="flex items-center gap-3 py-6">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <div>
              <div className="font-medium">載入雙 Agent 狀態失敗</div>
              <div className="mt-1 text-sm text-muted-foreground">{err}</div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!status) return null;

  const d = status.data;
  const sc = status.scorecard;
  const busOpen = bus.filter((t) => !["done", "fail", "timeout"].includes(t.status));
  const busRecent = bus.filter((t) => ["done", "fail", "timeout"].includes(t.status)).slice(0, 5);
  const hfPending = hf.filter((c) => c.status.includes("pending"));
  const hfAccepted = hf.filter((c) => c.status.startsWith("accepted"));

  const trendIcon = sc.trend.includes("📈") ? (
    <TrendingUp className="h-4 w-4 text-green-500" />
  ) : sc.trend.includes("📉") ? (
    <TrendingDown className="h-4 w-4 text-red-500" />
  ) : (
    <Activity className="h-4 w-4 text-muted-foreground" />
  );

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      {/* -------- Header -------- */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">雙 Agent 控制台</h1>
          <div className="mt-1 text-sm text-muted-foreground">
            Hermes × OpenClaw 合夥人即時狀態 · 自動每 30 秒刷新 ·{" "}
            <span className="font-mono">{d.ts}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Button onClick={() => setShowDispatch(true)} className="gap-2">
            <Send className="h-4 w-4" /> 派工給龍蝦
          </Button>
          <Button variant="outline" onClick={() => setShowCoaching(true)} className="gap-2">
            <MessageSquare className="h-4 w-4" /> Coaching
          </Button>
          <Button variant="outline" onClick={refresh} className="gap-2">
            <RefreshCw className="h-4 w-4" /> 刷新
          </Button>
          <div className="text-right">
            <div className="flex items-center justify-end gap-2 text-4xl font-bold">
              {sc.score}
              <span className="text-xl text-muted-foreground">/ {sc.max}</span>
            </div>
            <div className="mt-1 flex items-center justify-end gap-1 text-xs text-muted-foreground">
              {trendIcon}
              <span className="truncate max-w-md">{sc.trend}</span>
            </div>
          </div>
        </div>
      </div>

      {/* -------- Toast -------- */}
      {toast && (
        <div className="fixed bottom-6 right-6 z-50 rounded-md border border-border bg-card px-4 py-3 shadow-lg">
          {toast}
        </div>
      )}

      {/* -------- Dispatch modal -------- */}
      {showDispatch && (
        <DispatchModal
          onClose={() => setShowDispatch(false)}
          onDispatched={(tid) => {
            flashToast(`🚀 dispatched ${tid}`);
            setShowDispatch(false);
            refresh();
          }}
        />
      )}

      {/* -------- Coaching modal -------- */}
      {showCoaching && (
        <CoachingModal
          onClose={() => setShowCoaching(false)}
          onCommented={() => {
            flashToast("✅ comment posted");
            refresh();
          }}
        />
      )}

      {/* -------- Scorecard breakdown -------- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="h-4 w-4" /> Scorecard
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(sc.breakdown).map(([name, tuple]) => {
              const [emoji, reason] = tuple;
              return (
                <div
                  key={name}
                  className="flex items-start gap-3 rounded-md border border-border/60 bg-card/50 p-3"
                >
                  <div className="text-xl">{emoji}</div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium">{name}</div>
                    <div className="mt-0.5 text-xs text-muted-foreground">{reason}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* -------- §9 Throttle status -------- */}
      {throttle && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Zap className="h-4 w-4" /> §9 Codex dispatch throttle
              {throttle.enforcement_enabled ? (
                <Badge variant="success">enforced</Badge>
              ) : (
                <Badge variant="warning">off</Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 md:grid-cols-2">
              {Object.entries(throttle.pairs).map(([pair, info]) => (
                <div key={pair} className="rounded-md border border-border/60 bg-card/50 p-3 text-sm">
                  <div className="flex items-center gap-2">
                    {info.allowed ? (
                      <Badge variant="success">ready</Badge>
                    ) : (
                      <Badge variant="destructive">blocked</Badge>
                    )}
                    <span className="font-mono text-xs">{pair.replace("_to_", " → ")}</span>
                  </div>
                  {info.reason && (
                    <div className="mt-1 text-xs text-muted-foreground">{info.reason}</div>
                  )}
                  <div className="mt-1 text-xs text-muted-foreground font-mono">
                    {JSON.stringify(info.stats)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* -------- Snapshot history chart -------- */}
      {snapshots.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <LineChart className="h-4 w-4" /> Score history（最近 {Math.min(snapshots.length, 20)} 份快照）
            </CardTitle>
          </CardHeader>
          <CardContent>
            <SnapshotSparkline snapshots={snapshots} />
          </CardContent>
        </Card>
      )}

      {/* -------- Middleware chain status -------- */}
      {middleware && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Layers className="h-4 w-4" /> Middleware chain（S3）
              {middleware.master_enabled ? (
                <Badge variant="success">{middleware.master_mode}</Badge>
              ) : (
                <Badge variant="warning">off</Badge>
              )}
              <span className="text-xs text-muted-foreground ml-2">
                {middleware.entries.filter((e) => e.enabled).length}/{middleware.entries.length} enabled
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
              {middleware.entries.map((e) => (
                <div
                  key={e.name}
                  className="flex items-center gap-2 rounded-md border border-border/60 bg-card/50 p-2 text-sm"
                >
                  <span className="font-mono text-xs text-muted-foreground w-8">
                    #{e.order}
                  </span>
                  {e.enabled ? (
                    <Badge variant="success">on</Badge>
                  ) : (
                    <Badge variant="outline">off</Badge>
                  )}
                  <span className="flex-1 font-mono text-xs">{e.name}</span>
                  {e.critical && <Badge variant="destructive">critical</Badge>}
                </div>
              ))}
            </div>
            {middleware.entries.length === 0 && (
              <div className="text-sm text-muted-foreground">
                尚未註冊 middleware（core.complete_task 第一次被呼叫時會自動 register_defaults）
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* -------- Auto-memory facts (S4) -------- */}
      {autoMemory && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Brain className="h-4 w-4" /> Auto-memory facts（S4）
              {autoMemory.exists ? (
                <Badge variant="success">{autoMemory.count || 0} facts</Badge>
              ) : (
                <Badge variant="outline">尚未寫入</Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {autoMemory.error ? (
              <div className="text-sm text-destructive">{autoMemory.error}</div>
            ) : autoMemory.facts.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                還沒 auto-extracted facts — bus task close 時會觸發 MemoryExtractionMiddleware，
                30s debounce 後從對話抽取 fact 寫入 <span className="font-mono">{autoMemory.path}</span>。
              </div>
            ) : (
              <div className="space-y-1 max-h-72 overflow-y-auto">
                {autoMemory.facts.map((f) => (
                  <div
                    key={f.id}
                    className="flex items-start gap-2 rounded border border-border/60 bg-card/50 p-2 text-sm"
                  >
                    <Badge variant="outline" className="mt-0.5">{f.category}</Badge>
                    <div className="min-w-0 flex-1">
                      <div>{f.content}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        confidence {f.confidence.toFixed(2)} · source {f.source}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* -------- Recent activity feed -------- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <History className="h-4 w-4" /> 最近 24h 活動 ({activity.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {activity.length === 0 ? (
            <div className="text-sm text-muted-foreground">無活動</div>
          ) : (
            <div className="space-y-1.5 max-h-96 overflow-y-auto">
              {activity.map((e, i) => (
                <div key={i} className="flex items-start gap-2 text-sm">
                  <Badge variant="outline" className="mt-0.5 shrink-0">{e.kind}</Badge>
                  <div className="min-w-0 flex-1">
                    <div className="truncate">{e.title}</div>
                    {e.subtitle && (
                      <div className="truncate text-xs text-muted-foreground">{e.subtitle}</div>
                    )}
                  </div>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {timeAgoShort(e.ts)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* -------- Two agents side-by-side -------- */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* -------- Hermes -------- */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <span>🧭</span> Hermes（行政合夥人）
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm">
              <div className="text-muted-foreground">Last daily review</div>
              <div className="font-mono">
                {d.daily_review?.hermes_last_reviewed || "—"}
                {d.daily_review?.today_done && <Badge variant="success" className="ml-2">今天已跑</Badge>}
              </div>
            </div>
            <div className="text-sm">
              <div className="text-muted-foreground">Outbound evolution notes</div>
              <div>
                總計 <span className="font-mono">{d.evolution.total_outbound_hermes_to_openclaw}</span>
                {" · "}本週 <span className="font-mono">{d.evolution.this_week_outbound}</span>
              </div>
            </div>
            <div className="text-sm">
              <div className="text-muted-foreground">Coaching session</div>
              <div>
                Empty-chair 預測 <span className="font-mono">{d.coaching.empty_chair_filled}</span> 段
              </div>
            </div>
          </CardContent>
        </Card>

        {/* -------- OpenClaw -------- */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <span>🦞</span> OpenClaw（技術合夥人 · 龍蝦）
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm">
              <div className="text-muted-foreground">Gateway</div>
              <div>
                {d.openclaw.gateway_state === "running" ? (
                  <Badge variant="success">Running</Badge>
                ) : (
                  <Badge variant="destructive">{d.openclaw.gateway_state || "unknown"}</Badge>
                )}
                {" "}
                {d.openclaw.codex_app_server_running ? (
                  <Badge variant="success" className="ml-1">Codex ready</Badge>
                ) : (
                  <Badge variant="warning" className="ml-1">Codex off</Badge>
                )}
              </div>
            </div>
            <div className="text-sm">
              <div className="text-muted-foreground">Inbound evolution notes</div>
              <div>
                總計 <span className="font-mono">{d.evolution.total_inbound_openclaw_to_hermes}</span>
                {" · "}本週 <span className="font-mono">{d.evolution.this_week_inbound}</span>
                {!d.evolution.m2_symmetry_ok && (
                  <Badge variant="destructive" className="ml-2">M2 asymmetry</Badge>
                )}
              </div>
            </div>
            <div className="text-sm">
              <div className="text-muted-foreground">Coaching session</div>
              <div>
                {d.coaching.blanks_openclaw}{" "}
                {d.coaching.blanks_openclaw && d.coaching.blanks_openclaw > 0 ? (
                  <Badge variant="warning">待回填</Badge>
                ) : (
                  <Badge variant="success">已完成</Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* -------- Bus tasks -------- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Network className="h-4 w-4" /> Agent Bus（se-013 finalizer gate）
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {busOpen.length > 0 && (
            <div>
              <div className="mb-2 text-sm font-medium">進行中 ({busOpen.length})</div>
              <div className="space-y-2">
                {busOpen.map((t) => (
                  <div
                    key={t.task_id}
                    className="flex items-start gap-3 rounded-md border border-border/60 bg-card/50 p-3"
                  >
                    <Badge variant={statusBadgeVariant(t.status)}>{t.status}</Badge>
                    <div className="min-w-0 flex-1">
                      <div className="text-sm">
                        <span className="font-mono text-xs text-muted-foreground">
                          {t.task_id}
                        </span>
                        {" · "}
                        <span>{t.from_agent} → {t.to_agent}</span>
                        {" · "}
                        <span className="text-xs text-muted-foreground">
                          {timeAgoShort(t.created_at)} ago
                        </span>
                      </div>
                      <div className="mt-1 text-sm text-foreground/80">{t.goal}</div>
                      <div className="mt-2 flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => closeBusTask(t.task_id, "done")}
                        >
                          done
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => closeBusTask(t.task_id, "fail")}
                        >
                          fail
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => closeBusTask(t.task_id, "keep-alive")}
                        >
                          keep-alive
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {busOpen.length === 0 && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              無 open task · finalizer gate 保持乾淨
            </div>
          )}

          <div>
            <div className="mb-2 text-sm font-medium">最近結案 ({busRecent.length})</div>
            <div className="space-y-1">
              {busRecent.map((t) => (
                <div key={t.task_id} className="flex items-center gap-3 text-sm">
                  <Badge variant={statusBadgeVariant(t.status)}>{t.status}</Badge>
                  <span className="font-mono text-xs text-muted-foreground">{t.task_id}</span>
                  <span className="truncate text-foreground/70">{t.goal}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="grid gap-2 border-t border-border/50 pt-3 text-xs text-muted-foreground md:grid-cols-3">
            <div>7d events: <span className="font-mono">{JSON.stringify(d.bus.events_7d)}</span></div>
            <div>keep-alive 7d: <span className="font-mono">{d.bus.keep_alive_7d ?? 0}</span></div>
            <div>amend_learning 7d: <span className="font-mono">{d.bus.amend_learning_7d ?? 0}</span></div>
          </div>
        </CardContent>
      </Card>

      {/* -------- HF cards -------- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <GitBranch className="h-4 w-4" /> Handoff cards（se-012）
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {hfPending.length > 0 && (
            <div>
              <div className="mb-2 text-sm font-medium">Pending ({hfPending.length})</div>
              <div className="space-y-2">
                {hfPending.map((c) => (
                  <div
                    key={c.file}
                    className="flex items-start gap-3 rounded-md border border-border/60 bg-card/50 p-3"
                  >
                    <Badge variant={hfStatusBadge(c.status)}>{c.status}</Badge>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{c.title}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        {c.from_agent} → {c.to_agent} · freshness {c.freshness_ts || "—"}
                      </div>
                      <div className="mt-0.5 text-xs text-muted-foreground font-mono">{c.file}</div>
                      <div className="mt-2 flex gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => hfDecision(c.file, "accept")}
                        >
                          accept
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => hfDecision(c.file, "reject")}
                        >
                          reject
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => hfDecision(c.file, "request-clarification")}
                        >
                          request-clarification
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div>
            <div className="mb-2 text-sm font-medium">
              已關閉 ({hfAccepted.length})
            </div>
            <div className="space-y-1 text-xs text-muted-foreground">
              {hfAccepted.slice(0, 5).map((c) => (
                <div key={c.file} className="flex items-center gap-2">
                  <Badge variant="outline">{c.status}</Badge>
                  <span className="truncate">{c.title}</span>
                </div>
              ))}
              {hfAccepted.length > 5 && (
                <div className="italic">... 還有 {hfAccepted.length - 5} 張</div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* -------- Next actions -------- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <MessageSquare className="h-4 w-4" /> 建議下一步
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm">
            {sc.next_actions.map((a, i) => (
              <li key={i} className="flex items-start gap-2">
                <Circle className="mt-0.5 h-3 w-3 flex-shrink-0 fill-primary/40 text-primary" />
                <span>{a}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* -------- Sync bridge detail -------- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Clock className="h-4 w-4" /> Sync bridge（se-007）
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 text-sm">
          {d.sync_bridge.checks?.map((c) => (
            <div key={c.id} className="flex items-center gap-2">
              {c.ok ? (
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              ) : (
                <AlertCircle className="h-4 w-4 text-destructive" />
              )}
              <span className="font-mono text-xs">{c.id}</span>
              <span className="text-xs text-muted-foreground">{c.detail}</span>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

// -------- Snapshot Sparkline --------
function SnapshotSparkline({ snapshots }: { snapshots: Snapshot[] }) {
  const dataPoints = [...snapshots].reverse().filter((s) => s.score !== null) as Array<
    Snapshot & { score: number }
  >;
  if (dataPoints.length < 2) {
    return <div className="text-sm text-muted-foreground">需要至少 2 份快照才能畫趨勢</div>;
  }
  const width = 600;
  const height = 80;
  const max = 10;
  const min = 0;
  const dx = width / (dataPoints.length - 1);
  const points = dataPoints.map((p, i) => {
    const x = i * dx;
    const y = height - ((p.score - min) / (max - min)) * height;
    return `${x},${y}`;
  });
  const path = "M " + points.join(" L ");

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height + 20}`} className="w-full h-24">
        {/* gridline for 5/10 */}
        <line
          x1="0"
          y1={height - (5 / 10) * height}
          x2={width}
          y2={height - (5 / 10) * height}
          stroke="currentColor"
          strokeOpacity="0.15"
          strokeDasharray="4 4"
        />
        <path d={path} fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
        {dataPoints.map((p, i) => (
          <circle
            key={i}
            cx={i * dx}
            cy={height - ((p.score - min) / (max - min)) * height}
            r="3"
            fill="currentColor"
          />
        ))}
      </svg>
      <div className="mt-1 flex justify-between text-xs text-muted-foreground font-mono">
        <span>最舊 {dataPoints[0].score}/10</span>
        <span>最新 {dataPoints[dataPoints.length - 1].score}/10</span>
      </div>
    </div>
  );
}

// -------- Coaching Modal --------
function CoachingModal({
  onClose,
  onCommented,
}: {
  onClose: () => void;
  onCommented: () => void;
}) {
  const [data, setData] = useState<{
    markdown: string;
    blanks: number;
    prompts: string[];
    mtime: number;
  } | null>(null);
  const [comment, setComment] = useState("");
  const [section, setSection] = useState("General");
  const [actor, setActor] = useState("brian");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/dual-agent/coaching")
      .then((r) => r.json())
      .then((d) => {
        if (d.error) setErr(d.error);
        else setData(d);
      });
  }, []);

  async function submit() {
    if (!comment.trim()) {
      setErr("comment 不能空");
      return;
    }
    setSubmitting(true);
    setErr(null);
    try {
      const res = await fetch("/api/dual-agent/coaching/comment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actor, section, comment: comment.trim() }),
      });
      const j = await res.json();
      if (j.ok) {
        onCommented();
        setComment("");
      } else {
        setErr(j.error || "post failed");
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-3xl max-h-[90vh] flex flex-col rounded-lg border border-border bg-card shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-lg font-semibold">
            Coaching session
            {data && data.blanks > 0 && (
              <Badge variant="warning" className="ml-2">
                {data.blanks} blanks
              </Badge>
            )}
          </h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {err && (
            <div className="mb-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {err}
            </div>
          )}
          {data ? (
            <>
              <div className="mb-4">
                <div className="mb-1 text-xs text-muted-foreground">Prompts ({data.prompts.length})</div>
                <div className="flex flex-wrap gap-1">
                  {data.prompts.map((p, i) => (
                    <Badge
                      key={i}
                      variant="outline"
                      className="cursor-pointer"
                      onClick={() => {
                        const match = p.match(/Q\d+\.\d+/);
                        if (match) setSection(match[0]);
                      }}
                    >
                      {p.length > 30 ? p.slice(0, 30) + "…" : p}
                    </Badge>
                  ))}
                </div>
              </div>
              <pre className="whitespace-pre-wrap text-xs font-mono leading-relaxed bg-background/50 p-3 rounded border border-border/60 max-h-96 overflow-y-auto">
                {data.markdown.slice(0, 8000)}
                {data.markdown.length > 8000 && "\n\n...（truncated，完整內容於 wiki 檔案）"}
              </pre>
            </>
          ) : (
            <div className="text-sm text-muted-foreground">Loading…</div>
          )}
        </div>

        <div className="border-t border-border px-5 py-3 space-y-2">
          <div className="text-sm font-medium">Quick comment</div>
          <div className="grid grid-cols-2 gap-2">
            <select
              value={actor}
              onChange={(e) => setActor(e.target.value)}
              className="rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            >
              <option value="brian">brian</option>
              <option value="hermes">hermes</option>
              <option value="openclaw">openclaw</option>
            </select>
            <input
              type="text"
              value={section}
              onChange={(e) => setSection(e.target.value)}
              placeholder="section (e.g. Q2.1)"
              className="rounded-md border border-input bg-background px-3 py-1.5 text-sm"
            />
          </div>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="一句話回應 / 指令 / 修正…"
            rows={2}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          />
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={onClose} disabled={submitting}>
              關閉
            </Button>
            <Button onClick={submit} disabled={submitting || !comment.trim()}>
              {submitting ? "送出中…" : "附註到 session"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// -------- Dispatch Modal --------
function DispatchModal({
  onClose,
  onDispatched,
}: {
  onClose: () => void;
  onDispatched: (taskId: string) => void;
}) {
  const [goal, setGoal] = useState("");
  const [context, setContext] = useState("");
  const [criteria, setCriteria] = useState("");
  const [priority, setPriority] = useState("P2");
  const [deadline, setDeadline] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    if (!goal.trim()) {
      setErr("goal 不能空");
      return;
    }
    setSubmitting(true);
    setErr(null);
    const body = {
      from_agent: "hermes",
      to_agent: "openclaw",
      goal: goal.trim(),
      context: context.trim() || undefined,
      success_criteria: criteria.trim() || undefined,
      priority,
      deadline_minutes: deadline ? parseInt(deadline, 10) : undefined,
    };
    try {
      const res = await fetch("/api/dual-agent/bus/dispatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await res.json();
      if (j.ok) {
        onDispatched(j.task_id);
      } else {
        setErr(j.error || "dispatch failed");
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-2xl rounded-lg border border-border bg-card shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-lg font-semibold">派工給龍蝦（Hermes → OpenClaw）</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="space-y-4 p-5">
          <div>
            <label className="block text-sm font-medium mb-1">Goal *</label>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="一句話講要做什麼。例：寫本週 inbound *_evolution_*.md，主題 openclaw-outbound-hook"
              rows={2}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Context（可選）</label>
            <textarea
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="背景資訊、相關檔案、前情提要"
              rows={3}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Success criteria（可選）</label>
            <textarea
              value={criteria}
              onChange={(e) => setCriteria(e.target.value)}
              placeholder="完成條件，怎麼算做完"
              rows={2}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Priority</label>
              <select
                value={priority}
                onChange={(e) => setPriority(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="P0">P0 — 立即</option>
                <option value="P1">P1 — 今天</option>
                <option value="P2">P2 — 本週</option>
                <option value="P3">P3 — 有空</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Deadline（分鐘，選填）</label>
              <input
                type="number"
                value={deadline}
                onChange={(e) => setDeadline(e.target.value)}
                placeholder="例 4320 = 3 天"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              />
            </div>
          </div>
          {err && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {err}
            </div>
          )}
        </div>
        <div className="flex justify-end gap-2 border-t border-border px-5 py-3">
          <Button variant="outline" onClick={onClose} disabled={submitting}>
            取消
          </Button>
          <Button onClick={submit} disabled={submitting}>
            {submitting ? "派工中…" : "派出去"}
          </Button>
        </div>
      </div>
    </div>
  );
}
