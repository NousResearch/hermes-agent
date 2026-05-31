import { useEffect, useMemo, useState } from "react";
import {
  api,
  type AgentProfileInfo,
  type AgentProfileDetail,
  type ActiveAgentInfo,
} from "@/lib/api";

/**
 * Agents — read-only view of delegation ``agent_profiles``.
 *
 * Surfaces the named sub-agents that ``delegate_task(profile=...)`` can target.
 * Strictly read-only: no create/edit/delete, no platform/channel binding. This
 * is distinct from the Profiles page (which manages multi-instance HERMES_HOME
 * profiles). If the gateway build has no ``agent_profiles`` configured the list
 * is simply empty.
 *
 * Strings are intentionally plain (not i18n) — this is a fork-local feature;
 * localisation can be layered in if/when it lands upstream.
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

  return (
    <div className="page agents-page">
      <div className="page-header">
        <h1>Agents</h1>
      </div>

      <p className="agents-intro">
        Read-only view of the delegation agents (<code>agent_profiles</code>) this
        gateway can hand work to via <code>delegate_task</code>. This does not
        bind agents to platforms or channels.
      </p>

      {active.length > 0 && (
        <div className="agents-active">
          <h2>Live now ({active.length})</h2>
          <ul className="agents-active-list">
            {active.map((a, i) => (
              <li key={a.subagent_id || i} className="agents-active-item">
                <span className="agents-active-goal">
                  {a.goal || a.subagent_id || "(sub-agent)"}
                </span>
                <span className="agents-active-meta">
                  {a.model || "?"} · depth {a.depth ?? "?"} · {a.status || "running"}
                  {typeof a.tool_count === "number" ? ` · ${a.tool_count} tools` : ""}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="agents-controls">
        <input
          type="text"
          className="agents-search"
          placeholder="Search agents…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>
              {m === "all" ? "All models" : m}
            </option>
          ))}
        </select>
        <select value={toolsetFilter} onChange={(e) => setToolsetFilter(e.target.value)}>
          {toolsets.map((ts) => (
            <option key={ts} value={ts}>
              {ts === "all" ? "All toolsets" : ts}
            </option>
          ))}
        </select>
      </div>

      {loading && <div className="loading">Loading…</div>}
      {error && <div className="error-banner">{error}</div>}

      {!loading && !error && agents.length === 0 && (
        <div className="agents-empty">
          No agent profiles are configured. Add an <code>agent_profiles:</code>{" "}
          section to your config to define delegation agents.
        </div>
      )}

      {!loading && !error && agents.length > 0 && filtered.length === 0 && (
        <div className="agents-empty">No agents match your filters.</div>
      )}

      <div className="agents-grid">
        {filtered.map((a) => {
          const detail = expanded[a.name];
          const isOpen = detail !== undefined;
          const fullPrompt =
            isOpen && detail !== "loading"
              ? (detail as AgentProfileDetail).system_prompt
              : null;
          return (
            <div key={a.name} className="agent-card">
              <div className="agent-card-header">
                <span className="agent-name">{a.name}</span>
                <span className="agent-model">{a.model || "(default model)"}</span>
              </div>

              {a.description && <p className="agent-desc">{a.description}</p>}

              <div className="agent-toolsets">
                {a.toolsets.map((ts) => (
                  <span key={ts} className="agent-toolset-chip">
                    {ts}
                  </span>
                ))}
              </div>

              <div className="agent-meta">
                {typeof a.tool_count === "number" && (
                  <span>tools: {a.tool_count}</span>
                )}
                {typeof a.max_iterations === "number" && (
                  <span>max iter: {a.max_iterations}</span>
                )}
              </div>

              {a.warnings.length > 0 ? (
                <ul className="agent-warnings">
                  {a.warnings.map((w, i) => (
                    <li key={i} className="agent-warning">
                      ⚠ {w}
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="agent-valid">✓ valid</div>
              )}

              <pre className="agent-prompt-preview">
                {detail === "loading"
                  ? "Loading…"
                  : fullPrompt ?? a.system_prompt_preview}
              </pre>

              <div className="agent-card-actions">
                <button className="agent-btn" onClick={() => toggleExpand(a.name)}>
                  {isOpen ? "Collapse" : "Expand prompt"}
                </button>
                <button className="agent-btn" onClick={() => copySnippet(a.name)}>
                  {copied === a.name ? "Copied!" : "Copy delegate_task()"}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
