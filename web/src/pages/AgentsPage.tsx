import { useCallback, useEffect, useMemo, useState } from "react";
import type { KeyboardEvent } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  MessageSquare,
  X,
  ExternalLink,
  Plus,
  RefreshCw,
  Save,
  Send,
  Trash2,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  ManagedAgentEntry,
  AgentConsoleSession,
  ManagedAgentsResponse,
  ManagedModelEntry,
} from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { usePageHeader } from "@/contexts/usePageHeader";
import { timeAgo } from "@/lib/utils";

const PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

const FALLBACK_TRIGGERS = [
  "quota_exceeded",
  "rate_limited",
  "timeout",
  "server_error",
  "empty_final_content",
] as const;

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n || 0);
}

function formatCost(n: number): string {
  if (n >= 1) return `$${n.toFixed(2)}`;
  if (n > 0) return `$${n.toFixed(4)}`;
  return "$0";
}

function usageTotal(agent: ManagedAgentEntry): number {
  const u = agent.usage;
  return (
    (u.input_tokens || 0) +
    (u.output_tokens || 0) +
    (u.cache_read_tokens || 0) +
    (u.reasoning_tokens || 0)
  );
}

function statusTone(status: string): "success" | "warning" | "secondary" | "destructive" {
  const normalized = status.toLowerCase();
  if (normalized === "active") return "success";
  if (normalized === "experimental") return "warning";
  if (normalized === "deprecated") return "destructive";
  return "secondary";
}

function sourceTone(source: string): "success" | "warning" | "secondary" | "destructive" {
  if (source === "live") return "success";
  if (source === "cache" || source === "manual") return "warning";
  return "secondary";
}

function ModelLabel({ model }: { model?: ManagedModelEntry }) {
  if (!model) return <span className="text-muted-foreground">Unknown model</span>;
  return (
    <span className="min-w-0">
      <span className="block truncate font-medium">{model.model_ref}</span>
      <span className="block truncate text-[11px] text-muted-foreground">
        {model.provider} / {model.model}
      </span>
    </span>
  );
}

function SubscriptionCell({ model }: { model?: ManagedModelEntry }) {
  const sub = model?.subscription;
  if (!sub || sub.source === "unavailable") {
    return <span className="text-xs text-muted-foreground">No subscription data</span>;
  }
  const limits = [
    sub.five_hour_limit_usd ? `$${sub.five_hour_limit_usd}/5h` : null,
    sub.weekly_limit_usd ? `$${sub.weekly_limit_usd}/wk` : null,
    sub.monthly_limit_usd ? `$${sub.monthly_limit_usd}/mo` : null,
  ].filter(Boolean);
  const requestLimits = sub.request_limits;
  const requestLine = requestLimits?.requests_per_5h
    ? `${requestLimits.requests_per_5h.toLocaleString()}/5h · ${requestLimits.requests_per_week?.toLocaleString() ?? "?"}/wk · ${requestLimits.requests_per_month?.toLocaleString() ?? "?"}/mo`
    : null;
  return (
    <div className="min-w-[10rem] space-y-1 text-xs">
      <div className="flex items-center gap-1.5">
        <Badge tone={sourceTone(sub.source)}>{sub.source}</Badge>
        {sub.usage_percent !== null && sub.usage_percent !== undefined ? (
          <span>{sub.usage_percent.toFixed(1)}% used</span>
        ) : null}
      </div>
      <div className="text-muted-foreground">
        {limits.length ? limits.join(" · ") : "limit n/a"}
        {sub.expires_at ? ` · expires ${sub.expires_at}` : ""}
      </div>
      {requestLine ? (
        <div className="text-muted-foreground">est. requests {requestLine}</div>
      ) : requestLimits?.notes ? (
        <div className="line-clamp-2 text-[11px] text-muted-foreground">{requestLimits.notes}</div>
      ) : null}
      {sub.reset_at ? (
        <div className="text-muted-foreground">resets {sub.reset_at}</div>
      ) : null}
      {sub.error ? (
        <div className="line-clamp-2 text-[11px] text-amber-600 dark:text-amber-400">
          {sub.error}
        </div>
      ) : null}
    </div>
  );
}

function ModelStrategySummary({ agent }: { agent: ManagedAgentEntry }) {
  const strategy = agent.model_strategy || {};
  const mode = strategy.mode || (agent.editable ? "fixed" : "external");
  const chain = strategy.chain || [];
  if (mode === "fallback" && chain.length > 1) {
    return (
      <div className="mt-1 max-w-[18rem] text-[11px] text-muted-foreground">
        <Badge tone="outline">fallback chain</Badge>
        <div className="mt-1 truncate">{chain.join(" → ")}</div>
      </div>
    );
  }
  if (mode === "external") {
    return (
      <div className="mt-1 text-[11px] text-muted-foreground">
        <Badge tone="secondary">external CLI</Badge>
      </div>
    );
  }
  return (
    <div className="mt-1 text-[11px] text-muted-foreground">
      <Badge tone="secondary">fixed model</Badge>
    </div>
  );
}

function AgentModelSelect({
  agent,
  models,
  modelByRef,
  onSaved,
}: {
  agent: ManagedAgentEntry;
  models: ManagedModelEntry[];
  modelByRef: Map<string, ManagedModelEntry>;
  onSaved: () => void;
}) {
  const [selected, setSelected] = useState(agent.model_ref);
  const [strategyMode, setStrategyMode] = useState<"fixed" | "fallback">(
    agent.model_strategy?.mode === "fallback" ? "fallback" : "fixed",
  );
  const [chain, setChain] = useState<string[]>(() => {
    const existing = agent.model_strategy?.chain || [];
    return existing.length ? existing : [agent.model_ref];
  });
  const [fallbackOn, setFallbackOn] = useState<string[]>(() => agent.model_strategy?.fallback_on || []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const normalizedChain = useMemo(() => {
    const ordered = [selected, ...chain].filter(Boolean);
    return ordered.filter((ref, idx) => ordered.indexOf(ref) === idx);
  }, [chain, selected]);

  useEffect(() => {
    setSelected(agent.model_ref);
    setStrategyMode(agent.model_strategy?.mode === "fallback" ? "fallback" : "fixed");
    setChain((agent.model_strategy?.chain || []).length ? agent.model_strategy?.chain || [] : [agent.model_ref]);
    setFallbackOn(agent.model_strategy?.fallback_on || []);
    setError(null);
  }, [agent.agent_id, agent.model_ref, agent.model_strategy]);

  const saveStrategy = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.setManagedAgentModelStrategy(agent.agent_id, {
        mode: strategyMode,
        primary: selected,
        chain: strategyMode === "fallback" ? normalizedChain : [selected],
        fallback_on: strategyMode === "fallback" ? fallbackOn : undefined,
      });
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const setNormalizedChain = (nextChain: string[]) => {
    const unique = nextChain.filter(Boolean).filter((ref, idx, arr) => arr.indexOf(ref) === idx);
    if (unique[0]) setSelected(unique[0]);
    setChain(unique);
  };

  const updateChainAt = (index: number, value: string) => {
    const next = normalizedChain.map((item, idx) => (idx === index ? value : item));
    setNormalizedChain(next);
  };

  const moveChainItem = (index: number, direction: -1 | 1) => {
    const next = [...normalizedChain];
    const target = index + direction;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    setNormalizedChain(next);
  };

  if (!agent.editable) {
    return (
      <div className="min-w-[14rem] space-y-1">
        <div className="flex items-center gap-1.5 text-xs font-medium">
          <ExternalLink className="h-3.5 w-3.5" />
          External CLI default
        </div>
        <div className="text-[11px] text-muted-foreground">
          {agent.runtime || "external runtime"} controls this model.
        </div>
      </div>
    );
  }

  return (
    <div className="min-w-[17rem] space-y-1.5">
      <div className="flex gap-2">
        <select
          className="h-8 min-w-0 flex-1 border border-border bg-background px-2 text-xs"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={busy}
        >
          {models
            .filter((m) => m.status !== "deprecated" || m.model_ref === agent.model_ref)
            .map((m) => (
              <option key={m.model_ref} value={m.model_ref}>
                {m.model_ref} · {m.provider}/{m.model}
              </option>
            ))}
        </select>
      </div>
      <ModelLabel model={modelByRef.get(selected)} />
      <ModelStrategySummary agent={agent} />
      <div className="space-y-1 border-t border-border/60 pt-1.5">
        <div className="flex items-center gap-2">
          <select
            className="h-7 border border-border bg-background px-2 text-[11px]"
            value={strategyMode}
            onChange={(e) => setStrategyMode(e.target.value === "fallback" ? "fallback" : "fixed")}
            disabled={busy}
          >
            <option value="fixed">fixed</option>
            <option value="fallback">fallback</option>
          </select>
          <Button
            size="sm"
            className="h-7 px-2 text-[11px]"
            outlined
            disabled={busy || (strategyMode === "fallback" && normalizedChain.length < 2)}
            onClick={saveStrategy}
            prefix={busy ? <Spinner /> : <Save className="h-3 w-3" />}
          >
            Strategy
          </Button>
        </div>
        {strategyMode === "fallback" ? (
          <div className="space-y-1">
            {normalizedChain.map((ref, index) => (
              <div key={`${ref}-${index}`} className="flex items-center gap-1">
                <select
                  className="h-7 min-w-0 flex-1 border border-border bg-background px-2 text-[11px]"
                  value={ref}
                  onChange={(e) => updateChainAt(index, e.target.value)}
                  disabled={busy}
                >
                  {models
                    .filter((m) => m.status !== "deprecated" || m.model_ref === ref)
                    .map((m) => (
                      <option key={m.model_ref} value={m.model_ref}>
                        {index + 1}. {m.model_ref}
                      </option>
                    ))}
                </select>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground disabled:opacity-30"
                  disabled={busy || index === 0}
                  onClick={() => moveChainItem(index, -1)}
                  aria-label="Move model up"
                >
                  <ChevronUp className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground disabled:opacity-30"
                  disabled={busy || index === normalizedChain.length - 1}
                  onClick={() => moveChainItem(index, 1)}
                  aria-label="Move model down"
                >
                  <ChevronDown className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  className="text-muted-foreground hover:text-destructive disabled:opacity-30"
                  disabled={busy || normalizedChain.length <= 2}
                  onClick={() => setNormalizedChain(normalizedChain.filter((_, idx) => idx !== index))}
                  aria-label="Remove model"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            <Button
              size="sm"
              className="h-7 px-2 text-[11px]"
              outlined
              disabled={busy}
              onClick={() => {
                const nextRef = models.find((m) => !normalizedChain.includes(m.model_ref))?.model_ref;
                setNormalizedChain([...normalizedChain, nextRef || models[0]?.model_ref || selected]);
              }}
              prefix={<Plus className="h-3 w-3" />}
            >
              Add fallback
            </Button>
            <div className="grid grid-cols-2 gap-1 pt-1">
              {FALLBACK_TRIGGERS.map((trigger) => (
                <label key={trigger} className="flex items-center gap-1 text-[11px] text-muted-foreground">
                  <input
                    type="checkbox"
                    checked={fallbackOn.includes(trigger)}
                    disabled={busy}
                    onChange={(event) => {
                      setFallbackOn((current) => (
                        event.target.checked
                          ? [...current, trigger].filter((item, idx, arr) => arr.indexOf(item) === idx)
                          : current.filter((item) => item !== trigger)
                      ));
                    }}
                  />
                  {trigger}
                </label>
              ))}
            </div>
          </div>
        ) : null}
      </div>
      {error ? <div className="text-[11px] text-destructive">{error}</div> : null}
    </div>
  );
}

export default function AgentsPage() {
  const { setTitle } = usePageHeader();
  const [period, setPeriod] = useState<(typeof PERIODS)[number]>(PERIODS[1]);
  const [data, setData] = useState<ManagedAgentsResponse | null>(null);
  const [consoleSessions, setConsoleSessions] = useState<AgentConsoleSession[]>([]);
  const [activeConsoleSessionId, setActiveConsoleSessionId] = useState<string | null>(null);
  const [consoleAgent, setConsoleAgent] = useState<ManagedAgentEntry | null>(null);
  const [consolePrompt, setConsolePrompt] = useState("");
  const [consoleWorkspace, setConsoleWorkspace] = useState("");
  const [defaultWorkspace, setDefaultWorkspace] = useState("");
  const [consoleRisk, setConsoleRisk] = useState("R0");
  const [consoleBusy, setConsoleBusy] = useState(false);
  const [consoleError, setConsoleError] = useState<string | null>(null);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.getManagedAgents(period.days));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [period.days]);

  const loadConsoleSessions = useCallback(async (agentId?: string) => {
    try {
      const resp = await api.getAgentConsoleSessions({ agent_id: agentId, limit: 20 });
      setConsoleSessions(resp.sessions);
      if (resp.sessions.length > 0) {
        setActiveConsoleSessionId((current) => (
          current && resp.sessions.some((session) => session.session_id === current)
            ? current
            : resp.sessions[0].session_id
        ));
      } else {
        setActiveConsoleSessionId(null);
      }
    } catch {
      setConsoleSessions([]);
      setActiveConsoleSessionId(null);
    }
  }, []);

  useEffect(() => {
    setTitle("Agent Control");
    return () => setTitle(null);
  }, [setTitle]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    api.getStatus()
      .then((status) => {
        setDefaultWorkspace(status.hermes_home);
        setConsoleWorkspace((current) => current || status.hermes_home);
      })
      .catch(() => {
        setDefaultWorkspace((current) => current || "");
      });
  }, []);

  const activeConsoleSession = useMemo(
    () => consoleSessions.find((session) => session.session_id === activeConsoleSessionId) || null,
    [consoleSessions, activeConsoleSessionId],
  );

  const startConsoleSession = async (agent: ManagedAgentEntry) => {
    setConsoleBusy(true);
    setConsoleError(null);
    try {
      const session = await api.createAgentConsoleSession(agent.agent_id, {
        workspace: consoleWorkspace,
        risk_level: consoleRisk,
      });
      setConsoleSessions((prev) => [session, ...prev.filter((item) => item.session_id !== session.session_id)]);
      setActiveConsoleSessionId(session.session_id);
    } catch (e) {
      setConsoleError(e instanceof Error ? e.message : String(e));
    } finally {
      setConsoleBusy(false);
    }
  };

  const closeConsoleSession = async (sessionId: string) => {
    setConsoleBusy(true);
    setConsoleError(null);
    try {
      await api.deleteAgentConsoleSession(sessionId);
      setConsoleSessions((prev) => {
        const next = prev.filter((session) => session.session_id !== sessionId);
        if (activeConsoleSessionId === sessionId) {
          setActiveConsoleSessionId(next[0]?.session_id ?? null);
        }
        return next;
      });
    } catch (e) {
      setConsoleError(e instanceof Error ? e.message : String(e));
    } finally {
      setConsoleBusy(false);
    }
  };

  const sendConsoleMessage = async () => {
    if (!consoleAgent) return;
    let session = activeConsoleSession;
    const prompt = consolePrompt.trim();
    if (!prompt) return;
    setConsoleBusy(true);
    setConsoleError(null);
    try {
      if (!session) {
        session = await api.createAgentConsoleSession(consoleAgent.agent_id, {
          workspace: consoleWorkspace,
          risk_level: consoleRisk,
        });
        setConsoleSessions((prev) => [session!, ...prev]);
        setActiveConsoleSessionId(session.session_id);
      }
      const updated = await api.sendAgentConsoleMessage(session.session_id, {
        prompt,
        workspace: consoleWorkspace,
        risk_level: consoleRisk,
      });
      setConsolePrompt("");
      setConsoleSessions((prev) => [updated, ...prev.filter((item) => item.session_id !== updated.session_id)]);
      setActiveConsoleSessionId(updated.session_id);
      setConsoleWorkspace(updated.workspace || consoleWorkspace);
      setConsoleRisk(updated.risk_level || consoleRisk);
    } catch (e) {
      setConsoleError(e instanceof Error ? e.message : String(e));
    } finally {
      setConsoleBusy(false);
    }
  };

  const handlePromptKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendConsoleMessage();
    }
  };

  const models = data?.models ?? [];
  const agents = data?.agents ?? [];
  const modelByRef = useMemo(
    () => new Map(models.map((m) => [m.model_ref, m])),
    [models],
  );
  const editableCount = agents.filter((a) => a.editable).length;
  const externalCount = agents.length - editableCount;
  const sortedAgents = useMemo(
    () => [...agents].sort((a, b) => usageTotal(b) - usageTotal(a)),
    [agents],
  );

  const openConsole = (agent: ManagedAgentEntry) => {
    setConsoleAgent(agent);
    setConsoleError(null);
    setConsoleOpen(true);
    void loadConsoleSessions(agent.agent_id);
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          {PERIODS.map((p) => (
            <Button
              key={p.label}
              size="sm"
              outlined={period.days !== p.days}
              onClick={() => setPeriod(p)}
            >
              {p.label}
            </Button>
          ))}
        </div>
        <Button
          size="sm"
          outlined
          onClick={load}
          disabled={loading}
          prefix={loading ? <Spinner /> : <RefreshCw className="h-4 w-4" />}
        >
          Refresh
        </Button>
      </div>

      {error ? (
        <Card className="border-destructive/40">
          <CardContent className="flex items-center gap-2 py-4 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-3 md:grid-cols-4">
        <Card>
          <CardContent className="py-4">
            <Stats
              items={[
                { label: "Agents", value: String(agents.length) },
                { label: "Editable", value: String(editableCount) },
                { label: "External CLI", value: String(externalCount) },
                { label: "Est. Cost", value: formatCost(data?.totals.estimated_cost ?? 0) },
              ]}
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle className="text-base">Agent Model Assignments</CardTitle>
            <div className="text-xs text-muted-foreground">
              {formatTokens((data?.totals.input_tokens ?? 0) + (data?.totals.output_tokens ?? 0))} tokens ·{" "}
              {data?.totals.api_calls ?? 0} calls · attribution{" "}
              {data ? `${data.totals.agent_attributed_events}/${data.totals.agent_attributed_events + data.totals.agent_unknown_events}` : "0/0"}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading && !data ? (
            <div className="flex items-center gap-2 py-10 text-sm text-muted-foreground">
              <Spinner /> Loading agents...
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1100px] text-left text-sm">
                <thead className="border-b border-border text-xs text-muted-foreground">
                  <tr>
                    <th className="py-2 pr-4 font-medium">Agent</th>
                    <th className="py-2 pr-4 font-medium">Model</th>
                    <th className="py-2 pr-4 font-medium">Runtime</th>
                    <th className="py-2 pr-4 font-medium">Usage</th>
                    <th className="py-2 pr-4 font-medium">Subscription</th>
                    <th className="py-2 font-medium">Tools</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {sortedAgents.map((agent) => {
                    const model = modelByRef.get(agent.model_ref);
                    return (
                      <tr key={agent.agent_id} className="align-top">
                        <td className="py-3 pr-4">
                          <div className="max-w-[18rem] space-y-1">
                            <div className="font-medium">{agent.display_name}</div>
                            <div className="font-mono text-[11px] text-muted-foreground">{agent.agent_id}</div>
                            <div className="line-clamp-2 text-xs text-muted-foreground">
                              {agent.role_summary}
                            </div>
                          </div>
                        </td>
                        <td className="py-3 pr-4">
                          <AgentModelSelect
                            agent={agent}
                            models={models}
                            modelByRef={modelByRef}
                            onSaved={load}
                          />
                        </td>
                        <td className="py-3 pr-4">
                          <div className="space-y-1">
                            <Badge tone={agent.editable ? "success" : "secondary"}>
                              {agent.editable ? "managed" : "external"}
                            </Badge>
                            <div className="text-xs text-muted-foreground">
                              {agent.runtime || "native"}
                            </div>
                            {model ? (
                              <Badge tone={statusTone(model.status)}>{model.status}</Badge>
                            ) : null}
                          </div>
                        </td>
                        <td className="py-3 pr-4">
                          <div className="space-y-1 text-xs">
                            <div className="font-medium">{formatTokens(usageTotal(agent))} tokens</div>
                            <div className="text-muted-foreground">
                              {agent.usage.api_calls} calls · {agent.usage.runs} runs
                            </div>
                            {agent.usage.last_used_at ? (
                              <div className="text-muted-foreground">
                                {timeAgo(agent.usage.last_used_at)}
                              </div>
                            ) : null}
                          </div>
                        </td>
                        <td className="py-3 pr-4">
                          <SubscriptionCell model={model} />
                        </td>
                        <td className="py-3">
                          <div className="mb-2 flex max-w-[16rem] flex-wrap gap-1">
                            {agent.tools.map((tool) => (
                              <span
                                key={tool}
                                className="bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
                              >
                                {tool}
                              </span>
                            ))}
                          </div>
                          <Button
                            size="sm"
                            outlined={consoleAgent?.agent_id !== agent.agent_id || !consoleOpen}
                            disabled={!agent.editable}
                            onClick={() => openConsole(agent)}
                            prefix={<MessageSquare className="h-3.5 w-3.5" />}
                          >
                            Open Console
                          </Button>
                          {!agent.editable ? (
                            <div className="mt-1 text-[11px] text-muted-foreground">
                              External CLI only
                            </div>
                          ) : null}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {consoleOpen ? (
        <div className="fixed inset-0 z-50 bg-background/60 backdrop-blur-sm">
          <div className="ml-auto flex h-full w-full max-w-3xl flex-col border-l border-border bg-background shadow-2xl">
            <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 text-primary" />
                  <h2 className="text-base font-semibold">Agent Console</h2>
                </div>
                <div className="mt-1 text-sm text-muted-foreground">
                  {consoleAgent ? (
                    <>
                      Chatting with <span className="font-medium text-foreground">{consoleAgent.display_name}</span> · {consoleAgent.model_ref}
                    </>
                  ) : (
                    "Choose an editable agent to start a console session."
                  )}
                </div>
              </div>
              <Button
                size="sm"
                outlined
                onClick={() => setConsoleOpen(false)}
                prefix={<X className="h-3.5 w-3.5" />}
              >
                Close
              </Button>
            </div>

            <div className="border-b border-border px-5 py-3">
              <div className="flex items-center gap-2 overflow-x-auto pb-1">
                {consoleSessions.map((session) => (
                  <div
                    key={session.session_id}
                    className={`shrink-0 border px-3 py-1.5 text-xs ${
                      activeConsoleSessionId === session.session_id
                        ? "border-primary bg-primary/10 text-foreground"
                        : "border-border bg-muted/20 text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        className="block max-w-36 truncate"
                        onClick={() => {
                          setActiveConsoleSessionId(session.session_id);
                          setConsoleWorkspace(session.workspace || defaultWorkspace);
                          setConsoleRisk(session.risk_level || "R0");
                        }}
                      >
                        {session.title || session.display_name}
                      </button>
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-destructive"
                        disabled={consoleBusy}
                        onClick={(event) => {
                          event.stopPropagation();
                          void closeConsoleSession(session.session_id);
                        }}
                        aria-label={`Close ${session.title || session.display_name}`}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                ))}
                {consoleAgent ? (
                  <Button
                    size="sm"
                    outlined
                    disabled={consoleBusy}
                    onClick={() => void startConsoleSession(consoleAgent)}
                    prefix={consoleBusy ? <Spinner /> : <Plus className="h-3.5 w-3.5" />}
                  >
                    New Chat
                  </Button>
                ) : null}
              </div>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
              {!activeConsoleSession ? (
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  Start a new chat, then type your first message below.
                </div>
              ) : activeConsoleSession.messages.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                  No messages yet.
                </div>
              ) : (
                <div className="space-y-4">
                  {activeConsoleSession.messages.map((message) => (
                    <div
                      key={message.message_id}
                      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[82%] border px-3 py-2 text-sm ${
                          message.role === "user"
                            ? "border-primary/30 bg-primary/10"
                            : "border-border bg-muted/20"
                        }`}
                      >
                        <div className="mb-1 text-[11px] uppercase tracking-wide text-muted-foreground">
                          {message.role === "user" ? "You" : activeConsoleSession.display_name}
                          {message.status ? ` · ${message.status}` : ""}
                        </div>
                        <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
                        {message.duration_seconds !== undefined && message.duration_seconds !== null ? (
                          <div className="mt-2 text-[11px] text-muted-foreground">
                            {message.duration_seconds}s{message.api_calls ? ` · ${message.api_calls} calls` : ""}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ))}
                  {consoleBusy ? (
                    <div className="flex justify-start">
                      <div className="border border-border bg-muted/20 px-3 py-2 text-sm text-muted-foreground">
                        <Spinner /> Thinking...
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
            </div>

            <div className="border-t border-border p-4">
              <div className="mb-2 flex flex-wrap gap-2">
                <input
                  className="h-8 min-w-0 flex-1 border border-border bg-background px-2 text-xs"
                  value={consoleWorkspace}
                  onChange={(e) => setConsoleWorkspace(e.target.value)}
                  disabled={!consoleAgent || consoleBusy}
                />
                <select
                  className="h-8 border border-border bg-background px-2 text-xs"
                  value={consoleRisk}
                  onChange={(e) => setConsoleRisk(e.target.value)}
                  disabled={!consoleAgent || consoleBusy}
                >
                  {["R0", "R1", "R2", "R3", "R4"].map((risk) => <option key={risk}>{risk}</option>)}
                </select>
              </div>
              <div className="flex items-end gap-2">
                <textarea
                  className="max-h-36 min-h-16 flex-1 resize-none border border-border bg-background p-3 text-sm outline-none focus:border-primary"
                  placeholder="Message this agent..."
                  value={consolePrompt}
                  onChange={(e) => setConsolePrompt(e.target.value)}
                  onKeyDown={handlePromptKeyDown}
                  disabled={!consoleAgent || consoleBusy}
                  autoFocus
                />
                <Button
                  size="sm"
                  className="mb-0.5"
                  disabled={!consoleAgent || !consolePrompt.trim() || consoleBusy}
                  onClick={() => void sendConsoleMessage()}
                  prefix={consoleBusy ? <Spinner /> : <Send className="h-3.5 w-3.5" />}
                >
                  Send
                </Button>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                Press Enter to send. Shift+Enter inserts a new line.
              </div>
              {consoleError ? <div className="mt-2 text-xs text-destructive">{consoleError}</div> : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
