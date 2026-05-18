import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  Bot,
  Cpu,
  Package,
  Play,
  RadioTower,
  Sparkles,
  Terminal,
  Wrench,
} from "lucide-react";
import { api } from "@/lib/api";
import type { ModelInfoResponse, PaginatedSessions, SkillInfo, StatusResponse, ToolsetInfo } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useSystemActions } from "@/contexts/useSystemActions";
import { isDashboardEmbeddedChatEnabled } from "@/lib/dashboard-flags";

interface ControlData {
  status: StatusResponse;
  sessions: PaginatedSessions;
  skills: SkillInfo[];
  toolsets: ToolsetInfo[];
  modelInfo: ModelInfoResponse;
}

export default function ControlPage() {
  const [data, setData] = useState<ControlData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { setAfterTitle, setEnd } = usePageHeader();
  const { isBusy, runAction } = useSystemActions();
  const embeddedChat = isDashboardEmbeddedChatEnabled();

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    return Promise.all([
      api.getStatus(),
      api.getSessions(8, 0),
      api.getSkills(),
      api.getToolsets(),
      api.getModelInfo(),
    ])
      .then(([status, sessions, skills, toolsets, modelInfo]) => {
        setData({ status, sessions, skills, toolsets, modelInfo });
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    void Promise.resolve().then(refresh);
  }, [refresh]);

  useLayoutEffect(() => {
    setAfterTitle(
      <span className="whitespace-nowrap text-xs text-muted-foreground">
        CHRONOS agent dashboard
      </span>,
    );
    setEnd(
      <Button size="xs" ghost onClick={refresh} disabled={loading}>
        {loading ? <Spinner /> : "Refresh"}
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [loading, refresh, setAfterTitle, setEnd]);

  const stats = useMemo(() => {
    const sessions = data?.sessions.sessions ?? [];
    const toolCalls = sessions.reduce((sum, s) => sum + (s.tool_call_count ?? 0), 0);
    const enabledToolsets = (data?.toolsets ?? []).filter((t) => t.enabled);
    const enabledTools = new Set(enabledToolsets.flatMap((t) => t.tools));
    const enabledSkills = (data?.skills ?? []).filter((s) => s.enabled);
    const connectedPlatforms = Object.entries(data?.status.gateway_platforms ?? {})
      .filter(([, p]) => p.state === "connected" || p.state === "running")
      .map(([name]) => name);

    return {
      toolCalls,
      enabledToolsets,
      enabledTools,
      enabledSkills,
      connectedPlatforms,
      modelSummary: `${data?.modelInfo.provider ?? "unknown"} / ${data?.modelInfo.model ?? "unknown"}`,
    };
  }, [data]);

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <Card>
        <CardContent className="py-8 text-sm text-destructive">
          Could not load CHRONOS dashboard: {error}
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  return (
    <div className="flex flex-col gap-4">
      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard
          icon={RadioTower}
          label="Gateway"
          value={data.status.gateway_running ? "Running" : "Offline"}
          detail={
            stats.connectedPlatforms.length > 0
              ? stats.connectedPlatforms.join(", ")
              : "No connected platforms"
          }
          tone={data.status.gateway_running ? "success" : "warning"}
        />
        <MetricCard
          icon={Activity}
          label="Active sessions"
          value={String(data.status.active_sessions)}
          detail={`${data.sessions.total} total conversations indexed`}
        />
        <MetricCard
          icon={Cpu}
          label="Inference"
          value={data.modelInfo.model}
          detail={data.modelInfo.provider === "openai-codex" ? "OpenAI Codex locked for topology work" : stats.modelSummary}
          tone={data.modelInfo.provider === "openai-codex" ? "success" : "warning"}
        />
        <MetricCard
          icon={Wrench}
          label="Tools"
          value={String(stats.enabledTools.size)}
          detail={`${stats.enabledToolsets.length} enabled toolsets`}
        />
        <MetricCard
          icon={Package}
          label="Skills"
          value={String(stats.enabledSkills.length)}
          detail={`${data.skills.length} installed skills`}
        />
      </section>

      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm flex items-center gap-2">
            <Sparkles className="h-4 w-4" />
            Topology research stack
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 px-4 pb-4 text-sm text-muted-foreground lg:grid-cols-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-midground">Model substrate</p>
            <p className="mt-1">Keep frontier inference on <span className="font-mono text-midground">{stats.modelSummary}</span>; use its tool-calling bandwidth to interrogate Atlas geometry.</p>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-midground">Regulator</p>
            <p className="mt-1">Novelty gate, exclusion pressure, anchor lifecycle, and exhaustion ledger stay visible as operational constraints.</p>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-midground">Training target</p>
            <p className="mt-1">Generate curated trajectories for models trained on the CHRONOS stack to see topology, not just emit prose.</p>
          </div>
        </CardContent>
      </Card>

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
        <Card>
          <CardHeader className="py-3 px-4">
            <div className="flex items-center justify-between gap-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Bot className="h-4 w-4" />
                CHRONOS action feed
              </CardTitle>
              <Badge tone="secondary" className="text-[10px]">
                {stats.toolCalls} recent tool calls
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="px-4 pb-4">
            <div className="grid gap-2">
              {data.sessions.sessions.length === 0 ? (
                <p className="py-8 text-center text-sm text-muted-foreground">
                  No sessions yet. Start the embedded chat to see actions here.
                </p>
              ) : (
                data.sessions.sessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => navigate("/sessions")}
                    className="group text-left border border-border bg-muted/10 px-3 py-2.5 transition-colors hover:bg-muted/30"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium">
                          {session.title || "Untitled session"}
                        </div>
                        <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                          <span>{session.source ?? "unknown"}</span>
                          <span>•</span>
                          <span>{session.model ?? "model unknown"}</span>
                          <span>•</span>
                          <span>{timeAgo(session.last_active)}</span>
                        </div>
                      </div>
                      <div className="flex shrink-0 gap-1">
                        <Badge tone="outline" className="text-[10px]">
                          {session.message_count} msgs
                        </Badge>
                        <Badge tone={session.tool_call_count > 0 ? "warning" : "secondary"} className="text-[10px]">
                          {session.tool_call_count} tools
                        </Badge>
                      </div>
                    </div>
                    {session.preview && (
                      <p className="mt-2 line-clamp-2 text-xs text-muted-foreground/80">
                        {session.preview}
                      </p>
                    )}
                  </button>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader className="py-3 px-4">
              <CardTitle className="text-sm flex items-center gap-2">
                <Terminal className="h-4 w-4" />
                Interact
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-2 px-4 pb-4">
              <Button onClick={() => navigate(embeddedChat ? "/chat" : "/sessions")}>
                <Play className="h-4 w-4" />
                {embeddedChat ? "Open live agent chat" : "Open sessions"}
              </Button>
              <Button ghost onClick={() => navigate("/skills")}>Inspect tools & skills</Button>
              <Button ghost onClick={() => navigate("/logs")}>View runtime logs</Button>
              <Button
                ghost
                disabled={isBusy}
                onClick={() => void runAction("restart")}
              >
                Restart gateway
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="py-3 px-4">
              <CardTitle className="text-sm flex items-center gap-2">
                <Sparkles className="h-4 w-4" />
                Enabled capabilities
              </CardTitle>
            </CardHeader>
            <CardContent className="px-4 pb-4">
              <div className="mb-3 flex flex-wrap gap-1">
                {stats.enabledToolsets.slice(0, 12).map((toolset) => (
                  <Badge key={toolset.name} tone="success" className="text-[10px] font-mono">
                    {toolset.name}
                  </Badge>
                ))}
                {stats.enabledToolsets.length > 12 && (
                  <Badge tone="secondary" className="text-[10px]">
                    +{stats.enabledToolsets.length - 12} more
                  </Badge>
                )}
              </div>
              <div className="flex flex-wrap gap-1">
                {stats.enabledSkills.slice(0, 14).map((skill) => (
                  <Badge key={skill.name} tone="outline" className="text-[10px] font-mono">
                    {skill.name}
                  </Badge>
                ))}
                {stats.enabledSkills.length > 14 && (
                  <Badge tone="secondary" className="text-[10px]">
                    +{stats.enabledSkills.length - 14} more skills
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {error && (
        <p className="text-xs text-warning">Last refresh failed: {error}</p>
      )}
    </div>
  );
}

function MetricCard({
  detail,
  icon: Icon,
  label,
  tone = "secondary",
  value,
}: {
  detail: string;
  icon: typeof Activity;
  label: string;
  tone?: "secondary" | "success" | "warning" | "outline";
  value: string;
}) {
  return (
    <Card>
      <CardContent className="px-4 py-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
              {label}
            </p>
            <p className="mt-1 text-2xl font-semibold tracking-tight">{value}</p>
          </div>
          <Badge tone={tone} className="p-2">
            <Icon className="h-4 w-4" />
          </Badge>
        </div>
        <p className="mt-2 truncate text-xs text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  );
}
