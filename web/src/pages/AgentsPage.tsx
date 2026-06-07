import { useEffect, useMemo, useState } from "react";
import {
  api,
  type AgentProfileInfo,
  type AgentProfileDetail,
  type ActiveAgentInfo,
} from "@/lib/api";
import { Bot } from "lucide-react";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";

/**
 * Agents — read-only view of delegation ``agent_profiles``.
 *
 * Surfaces the named sub-agents that ``delegate_task(profile=...)`` can target.
 * Strictly read-only: no create/edit/delete, no platform/channel binding. This
 * is distinct from the Profiles page (which manages multi-instance HERMES_HOME
 * profiles). If the gateway build has no ``agent_profiles`` configured the list
 * is simply empty.
 *
 * Styling follows the dashboard idiom: H2 for the page title, Tailwind theme
 * utility classes (text-muted-foreground, text-destructive, bg-muted, etc.)
 * for body text, and the shared Card / Button / Input primitives — so it
 * inherits the active theme's colors automatically.
 */
export default function AgentsPage() {
  const [agents, setAgents] = useState<AgentProfileInfo[]>([]);
  const [active, setActive] = useState<ActiveAgentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [modelFilter, setModelFilter] = useState("all");
  const [toolsetFilter, setToolsetFilter] = useState("all");
  const [expanded, setExpanded] = useState<
    Record<string, AgentProfileDetail | "loading" | undefined>
  >({});
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await api.getAgentProfiles();
        if (!cancelled) setAgents(data.agent_profiles || []);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Poll active sub-agents every 5s (best-effort; empty when nothing running).
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const data = await api.getActiveAgents();
        if (!cancelled) setActive(data.active || []);
      } catch {
        /* non-fatal: keep previous value */
      }
    };
    poll();
    const timer = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const models = useMemo(() => {
    const set = new Set<string>();
    for (const a of agents) if (a.model) set.add(a.model);
    return ["all", ...Array.from(set).sort()];
  }, [agents]);

  const toolsets = useMemo(() => {
    const set = new Set<string>();
    for (const a of agents) for (const ts of a.toolsets) set.add(ts);
    return ["all", ...Array.from(set).sort()];
  }, [agents]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return agents.filter((a) => {
      if (modelFilter !== "all" && a.model !== modelFilter) return false;
      if (toolsetFilter !== "all" && !a.toolsets.includes(toolsetFilter)) return false;
      if (!q) return true;
      return (
        a.name.toLowerCase().includes(q) ||
        a.model.toLowerCase().includes(q) ||
        a.description.toLowerCase().includes(q) ||
        a.toolsets.some((ts) => ts.toLowerCase().includes(q))
      );
    });
  }, [agents, search, modelFilter, toolsetFilter]);

  const toggleExpand = async (name: string) => {
    if (expanded[name]) {
      setExpanded((prev) => ({ ...prev, [name]: undefined }));
      return;
    }
    setExpanded((prev) => ({ ...prev, [name]: "loading" }));
    try {
      const detail = await api.getAgentProfile(name);
      setExpanded((prev) => ({ ...prev, [name]: detail }));
    } catch {
      setExpanded((prev) => ({ ...prev, [name]: undefined }));
    }
  };

  const copySnippet = (name: string) => {
    const snippet = `delegate_task(profile="${name}", task="...")`;
    try {
      navigator.clipboard?.writeText(snippet);
      setCopied(name);
      setTimeout(() => setCopied((c) => (c === name ? null : c)), 1500);
    } catch {
      /* clipboard unavailable — no-op */
    }
  };

  const selectClasses =
    "h-9 rounded-md border border-input bg-background px-2 text-sm " +
    "text-foreground focus:outline-none focus:ring-2 focus:ring-ring";

  if (loading) {
    return (
      <div className="flex flex-col gap-6">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Bot className="h-4 w-4" />
          Agents ({agents.length})
        </H2>
        <p className="text-xs text-muted-foreground">
          Read-only view of the delegation agents (<code>agent_profiles</code>) this
          gateway can hand work to via <code>delegate_task</code>. Does not bind
          agents to platforms or channels.
        </p>
      </div>

      {error && (
        <Card className="border-destructive/40">
          <CardContent className="py-4">
            <p className="text-sm text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}

      {active.length > 0 && (
        <Card>
          <CardContent className="py-4 flex flex-col gap-2">
            <span className="text-sm font-bold uppercase tracking-wide">
              Live now ({active.length})
            </span>
            <div className="flex flex-col gap-1">
              {active.map((a, i) => (
                <div
                  key={a.subagent_id || i}
                  className="flex items-baseline justify-between gap-3"
                >
                  <span className="text-sm truncate">
                    {a.goal || a.subagent_id || "(sub-agent)"}
                  </span>
                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                    {a.model || "?"} · depth {a.depth ?? "?"} · {a.status || "running"}
                    {typeof a.tool_count === "number" ? ` · ${a.tool_count} tools` : ""}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* controls */}
      <div className="flex flex-wrap items-center gap-2">
        <Input
          type="text"
          placeholder="Search agents…"
          value={search}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
          className="max-w-xs"
        />
        <select
          className={selectClasses}
          value={modelFilter}
          onChange={(e) => setModelFilter(e.target.value)}
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {m === "all" ? "All models" : m}
            </option>
          ))}
        </select>
        <select
          className={selectClasses}
          value={toolsetFilter}
          onChange={(e) => setToolsetFilter(e.target.value)}
        >
          {toolsets.map((ts) => (
            <option key={ts} value={ts}>
              {ts === "all" ? "All toolsets" : ts}
            </option>
          ))}
        </select>
      </div>

      {agents.length === 0 && (
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            No agent profiles are configured. Add an <code>agent_profiles:</code>{" "}
            section to your config to define delegation agents.
          </CardContent>
        </Card>
      )}

      {agents.length > 0 && filtered.length === 0 && (
        <p className="text-sm text-muted-foreground">No agents match your filters.</p>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        {filtered.map((a) => {
          const detail = expanded[a.name];
          const isOpen = detail !== undefined;
          const fullPrompt =
            isOpen && detail !== "loading"
              ? (detail as AgentProfileDetail).system_prompt
              : null;
          return (
            <Card key={a.name} className="transition-colors hover:border-primary/40">
              <CardContent className="py-4 flex flex-col gap-3">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-base font-bold truncate">{a.name}</span>
                  <span className="text-xs text-muted-foreground truncate whitespace-nowrap font-mono">
                    {a.model || "(default model)"}
                  </span>
                </div>

                {a.description && (
                  <p className="text-xs text-muted-foreground">{a.description}</p>
                )}

                <div className="flex flex-wrap gap-1.5">
                  {a.toolsets.map((ts) => (
                    <span
                      key={ts}
                      className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary font-mono"
                    >
                      {ts}
                    </span>
                  ))}
                </div>

                <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                  {typeof a.tool_count === "number" && <span>tools: {a.tool_count}</span>}
                  {typeof a.max_iterations === "number" && (
                    <span>max iter: {a.max_iterations}</span>
                  )}
                </div>

                {a.warnings.length > 0 ? (
                  <div className="flex flex-col gap-1">
                    {a.warnings.map((w, i) => (
                      <Badge key={i} tone="warning" className="w-fit">
                        ⚠ {w}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <Badge tone="success" className="w-fit">
                    valid
                  </Badge>
                )}

                <pre className="text-xs whitespace-pre-wrap break-words rounded bg-muted p-2 text-muted-foreground max-h-64 overflow-auto font-mono">
                  {detail === "loading"
                    ? "Loading…"
                    : fullPrompt ?? a.system_prompt_preview}
                </pre>

                <div className="flex gap-2">
                  <Button outlined size="sm" onClick={() => toggleExpand(a.name)}>
                    {isOpen ? "Collapse" : "Expand prompt"}
                  </Button>
                  <Button outlined size="sm" onClick={() => copySnippet(a.name)}>
                    {copied === a.name ? "Copied!" : "Copy delegate_task()"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
