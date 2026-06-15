import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Brain,
  ChevronDown,
  Cpu,
  DollarSign,
  Eye,
  RefreshCw,
  Settings2,
  Star,
  Wrench,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  AuxiliaryModelsResponse,
  AuxiliaryTaskAssignment,
  FallbackEntry,
  ModelsAnalyticsModelEntry,
  ModelsAnalyticsResponse,
} from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { formatTokenCount } from "@/lib/format";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";
import { ModelPickerDialog } from "@/components/ModelPickerDialog";
import { ModelReloadConfirm } from "@/components/ModelReloadConfirm";
import { TabsList, TabsTrigger } from "@nous-research/ui/ui/components/tabs";

const VALID_TABS = new Set(["main-model", "auxiliary-tasks", "used-models"]);

function getTabFromQuery(searchParams: URLSearchParams): string {
  const tab = searchParams.get("tab");
  if (tab && VALID_TABS.has(tab)) return tab;
  return "main-model";
}

const PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

// Must match _AUX_TASK_SLOTS in hermes_cli/web_server.py.
const AUX_TASKS: readonly { key: string; label: string; hint: string }[] = [
  { key: "vision", label: "Vision", hint: "Image analysis" },
  { key: "web_extract", label: "Web Extract", hint: "Page summarization" },
  { key: "compression", label: "Compression", hint: "Context compaction" },
  { key: "skills_hub", label: "Skills Hub", hint: "Skill search" },
  { key: "approval", label: "Approval", hint: "Smart auto-approve" },
  { key: "mcp", label: "MCP", hint: "MCP tool routing" },
  { key: "title_generation", label: "Title Gen", hint: "Session titles" },
  { key: "triage_specifier", label: "Triage Specifier", hint: "Kanban spec fleshing" },
  { key: "kanban_decomposer", label: "Kanban Decomposer", hint: "Task decomposition" },
  { key: "profile_describer", label: "Profile Describer", hint: "Auto profile descriptions" },
  { key: "curator", label: "Curator", hint: "Skill-usage review" },
] as const;

const AUX_TASK_METADATA: Record<string, { label: string; hint: string }> = {
  vision: { label: "Vision", hint: "Image analysis" },
  web_extract: { label: "Web Extract", hint: "Page summarization" },
  compression: { label: "Compression", hint: "Context compaction" },
  session_search: { label: "Session Search", hint: "Recall queries" },
  skills_hub: { label: "Skills Hub", hint: "Skill search" },
  approval: { label: "Approval", hint: "Smart auto-approve" },
  mcp: { label: "MCP", hint: "MCP tool routing" },
  title_generation: { label: "Title Gen", hint: "Session titles" },
  triage_specifier: { label: "Triage Specifier", hint: "Kanban spec fleshing" },
  kanban_decomposer: { label: "Kanban Decomposer", hint: "Task decomposition" },
  profile_describer: { label: "Profile Describer", hint: "Auto profile descriptions" },
  curator: { label: "Curator", hint: "Skill-usage review" },
};

function formatAuxTaskLabel(task: string): string {
  return task.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function getAuxTaskMetadata(task: string): { label: string; hint: string } {
  return AUX_TASK_METADATA[task] ?? { label: formatAuxTaskLabel(task), hint: "Auxiliary model task" };
}

function fallbackEntryKey(entry: FallbackEntry): string {
  return [entry.provider, entry.model, entry.base_url ?? "", entry.api_mode ?? ""].join("\u001f");
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(n: number): string {
  if (n >= 1) return `$${n.toFixed(2)}`;
  if (n >= 0.01) return `$${n.toFixed(3)}`;
  if (n > 0) return `$${n.toFixed(4)}`;
  return "$0";
}

function shortModelName(model: string): string {
  const slashIdx = model.indexOf("/");
  if (slashIdx > 0) return model.slice(slashIdx + 1);
  return model;
}

function modelVendor(model: string, fallback?: string): string {
  const slashIdx = model.indexOf("/");
  if (slashIdx > 0) return model.slice(0, slashIdx);
  return fallback || "";
}

function TokenBar({
  input, output, cacheRead, reasoning,
}: { input: number; output: number; cacheRead: number; reasoning: number }) {
  const total = input + output + cacheRead + reasoning;
  if (total === 0) return null;

  // Segments carry a CSS color value (hex or `var(--token)`) rather than
  // a Tailwind class so the input/output series can pick up the active
  // theme's `--series-*-token` vars — see `themes/types.ts`
  // `ThemeSeriesColors`. The /60–/70 fade on the bar is applied via
  // color-mix on the same value so themes don't need to ship two
  // separate hex literals.
  const segments: Array<{ color: string; label: string; value: number }> = [
    { value: cacheRead, color: "#60a5fa", label: "Cache Read" }, // tailwind blue-400
    { value: reasoning, color: "#c084fc", label: "Reasoning" }, // tailwind purple-400
    { value: input, color: "var(--series-input-token)", label: "Input" },
    { value: output, color: "var(--series-output-token)", label: "Output" },
  ].filter((s) => s.value > 0);

  return (
    <div className="space-y-1.5">
      <div className="relative flex min-h-[1.5rem] w-full items-stretch overflow-hidden">
        {segments.map((s, i) => (
          <div
            key={i}
            className="relative flex items-center transition-all duration-300"
            style={{
              backgroundColor: `color-mix(in srgb, ${s.color} 70%, transparent)`,
              width: `${(s.value / total) * 100}%`,
            }}
          >
            <div
              className="absolute inset-0 opacity-30"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(to right, transparent 0 0.4rem, currentColor 0.4rem calc(0.4rem + 1px))",
              }}
            />
          </div>
        ))}
      </div>
      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-text-secondary">
        {segments.map((s, i) => (
          <span key={i} className="flex items-center gap-1">
            <span
              className="inline-block h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: s.color }}
            />
            {s.label} {formatTokens(s.value)}
          </span>
        ))}
      </div>
    </div>
  );
}

function CapabilityBadges({
  capabilities,
}: { capabilities: ModelsAnalyticsModelEntry["capabilities"] }) {
  const hasAny =
    capabilities.supports_tools ||
    capabilities.supports_vision ||
    capabilities.supports_reasoning ||
    capabilities.model_family;
  if (!hasAny) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {capabilities.supports_tools && (
        <span className="inline-flex items-center gap-1 bg-success/10 px-1.5 py-0.5 text-xs font-medium text-success">
          <Wrench className="h-2.5 w-2.5" /> Tools
        </span>
      )}
      {capabilities.supports_vision && (
        <span className="inline-flex items-center gap-1 bg-blue-500/10 px-1.5 py-0.5 text-xs font-medium text-blue-600 dark:text-blue-400">
          <Eye className="h-2.5 w-2.5" /> Vision
        </span>
      )}
      {capabilities.supports_reasoning && (
        <span className="inline-flex items-center gap-1 bg-purple-500/10 px-1.5 py-0.5 text-xs font-medium text-purple-600 dark:text-purple-400">
          <Brain className="h-2.5 w-2.5" /> Reasoning
        </span>
      )}
      {capabilities.model_family && (
        <span className="inline-flex items-center bg-muted px-1.5 py-0.5 text-xs font-medium text-text-secondary">
          {capabilities.model_family}
        </span>
      )}
    </div>
  );
}

/* ─── Per-card "Use as" menu ─── */

function UseAsMenu({
  provider, model, isMain, mainAuxTask, onAssigned,
}: {
  provider: string; model: string;
  isMain: boolean;
  mainAuxTask: string | null;
  onAssigned(): void;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingConfirm, setPendingConfirm] = useState<{
    message: string;
    scope: "main" | "auxiliary";
    task: string;
  } | null>(null);

  const assign = async (
    scope: "main" | "auxiliary",
    task: string,
    confirmExpensiveModel = false,
  ) => {
    if (!provider || !model) {
      setError("Missing provider/model");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await api.setModelAssignment({
        confirm_expensive_model: confirmExpensiveModel,
        scope,
        provider,
        model,
        task,
      });
      if (result.confirm_required) {
        setPendingConfirm({
          scope,
          task,
          message:
            result.confirm_message ||
            "This model has unusually high known pricing.",
        });
        return;
      }
      onAssigned();
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && !target.closest?.("[data-use-as-menu]")) setOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [open]);

  return (
    <div className="relative" data-use-as-menu>
      <Button
        size="sm"
        outlined
        onClick={() => setOpen((v) => !v)}
        disabled={busy}
        className="h-6 px-2 text-xs uppercase"
        prefix={busy ? <Spinner /> : null}
        data-testid="used-model-use-as-button"
      >
        Use as <ChevronDown className="h-3 w-3" />
      </Button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 min-w-[220px] border border-border bg-card shadow-lg" data-testid="used-model-use-as-menu">
          <button
            type="button"
            onClick={() => assign("main", "")}
            disabled={busy}
            className="flex w-full items-center justify-between px-3 py-2 text-xs uppercase hover:bg-muted/50 disabled:opacity-40"
          >
            <span className="flex items-center gap-2">
              <Star className="h-3 w-3" />
              Main model
            </span>
            {isMain && (
              <span className="text-display text-xs tracking-wider text-primary">
                current
              </span>
            )}
          </button>

          <div className="border-t border-border/50 px-3 py-1.5 text-display text-xs tracking-wider text-text-tertiary">
            Auxiliary task
          </div>

          <button
            type="button"
            onClick={() => assign("auxiliary", "")}
            disabled={busy}
            className="flex w-full items-center justify-between px-3 py-1.5 text-xs uppercase hover:bg-muted/50 disabled:opacity-40"
          >
            <span>All auxiliary tasks</span>
          </button>

          {AUX_TASKS.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => assign("auxiliary", t.key)}
              disabled={busy}
              className="flex w-full items-center justify-between px-3 py-1.5 text-xs uppercase hover:bg-muted/50 disabled:opacity-40"
            >
              <span>{t.label}</span>
              {mainAuxTask === t.key && (
                <span className="text-display text-xs tracking-wider text-primary">
                  current
                </span>
              )}
            </button>
          ))}

          {error && (
            <div className="px-3 py-2 text-xs text-destructive border-t border-border/50">
              {error}
            </div>
          )}
        </div>
      )}
      <ConfirmDialog
        open={!!pendingConfirm}
        title="Expensive Model Warning"
        description={pendingConfirm?.message}
        destructive
        confirmLabel="Switch anyway"
        cancelLabel="Cancel"
        loading={busy}
        onCancel={() => setPendingConfirm(null)}
        onConfirm={() => {
          const pending = pendingConfirm;
          if (!pending) return;
          setPendingConfirm(null);
          void assign(pending.scope, pending.task, true);
        }}
      />
    </div>
  );
}

/* ─── ModelCard ─── */

function ModelCard({
  entry, rank, main, aux, onAssigned, showTokens,
}: {
  entry: ModelsAnalyticsModelEntry; rank: number;
  main: { provider: string; model: string } | null;
  aux: AuxiliaryTaskAssignment[];
  onAssigned(): void; showTokens: boolean;
}) {
  const { t } = useI18n();
  const provider = entry.provider || modelVendor(entry.model);
  const totalTokens = entry.input_tokens + entry.output_tokens;

  const isMain = !!main && main.provider === provider && main.model === entry.model;
  const mainAuxTask = aux.find((a) => a.provider === provider && a.model === entry.model)?.task ?? null;

  return (
    <Card className={isMain ? "ring-1 ring-primary/40" : undefined} data-testid="used-model-card">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-text-tertiary text-xs font-mono">
                #{rank}
              </span>
              <CardTitle className="text-sm font-mono-ui truncate">
                {shortModelName(entry.model)}
              </CardTitle>
              {isMain && (
                <span className="inline-flex items-center gap-0.5 bg-primary/15 px-1.5 py-0.5 text-display text-xs font-medium tracking-wider text-primary">
                  <Star className="h-2.5 w-2.5" /> main
                </span>
              )}
              {mainAuxTask && (
                <span className="inline-flex items-center bg-purple-500/10 px-1.5 py-0.5 text-display text-xs font-medium tracking-wider text-purple-600 dark:text-purple-400">
                  aux · {mainAuxTask}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-1">
              {provider && (
                <Badge tone="secondary" className="text-xs">
                  {provider}
                </Badge>
              )}
              {entry.capabilities.context_window && entry.capabilities.context_window > 0 && (
                <span className="text-xs text-text-secondary">
                  {formatTokenCount(entry.capabilities.context_window)} ctx
                </span>
              )}
              {entry.capabilities.max_output_tokens && entry.capabilities.max_output_tokens > 0 && (
                <span className="text-xs text-text-secondary">
                  {formatTokenCount(entry.capabilities.max_output_tokens)} out
                </span>
              )}
            </div>
          </div>
          <div className="flex flex-col items-end gap-1 shrink-0">
            {showTokens ? (
              <div className="text-right">
                <div className="text-xs font-mono font-semibold">
                  {formatTokens(totalTokens)}
                </div>
                <div className="text-xs text-text-tertiary">
                  {t.models.tokens}
                </div>
              </div>
            ) : (
              entry.sessions > 0 && (
                <div className="text-right">
                  <div className="text-xs font-mono font-semibold">
                    {entry.sessions}
                  </div>
                  <div className="text-xs text-text-tertiary">
                    {t.models.sessions}
                  </div>
                </div>
              )
            )}
            <UseAsMenu
              provider={provider}
              model={entry.model}
              isMain={isMain}
              mainAuxTask={mainAuxTask}
              onAssigned={onAssigned}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-3">
        {showTokens && (
          <>
            <TokenBar input={entry.input_tokens} output={entry.output_tokens} cacheRead={entry.cache_read_tokens} reasoning={entry.reasoning_tokens} />
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="text-center">
                <div className="font-mono font-semibold">{entry.sessions}</div>
                <div className="text-xs text-text-tertiary">
                  {t.models.sessions}
                </div>
              </div>
              <div className="text-center">
                <div className="font-mono font-semibold">
                  {formatTokens(entry.avg_tokens_per_session)}
                </div>
                <div className="text-xs text-text-tertiary">
                  {t.models.avgPerSession}
                </div>
              </div>
              <div className="text-center">
                <div className="font-mono font-semibold">
                  {entry.api_calls > 0 ? formatTokens(entry.api_calls) : "—"}
                </div>
                <div className="text-xs text-text-tertiary">
                  {t.models.apiCalls}
                </div>
              </div>
            </div>
          </>
        )}

        <div className="flex items-center justify-between text-xs text-text-secondary border-t border-border/30 pt-2">
          <div className="flex items-center gap-3">
            {showTokens && entry.estimated_cost > 0 && <span className="flex items-center gap-0.5"><DollarSign className="h-2.5 w-2.5" />{formatCost(entry.estimated_cost)}</span>}
            {showTokens && entry.tool_calls > 0 && <span className="flex items-center gap-0.5"><Zap className="h-2.5 w-2.5" />{entry.tool_calls} {t.models.toolCalls}</span>}
          </div>
          {entry.last_used_at > 0 && <span>{timeAgo(entry.last_used_at)}</span>}
        </div>
        <CapabilityBadges capabilities={entry.capabilities} />
      </CardContent>
    </Card>
  );
}

/* ─── AuxiliaryTasksPanel (inline) ─── */

type PickerTarget = { kind: "main" } | { kind: "aux"; task: string } | { kind: "fallback" };

function AuxiliaryTasksPanel({
  aux, refreshKey, onSaved,
}: { aux: AuxiliaryModelsResponse | null; refreshKey: number; onSaved(): void }) {
  const [picker, setPicker] = useState<PickerTarget | null>(null);

  return (
    <>
      <div className="space-y-1 p-6 border border-border/50 rounded-lg bg-card/30">
        {(aux?.tasks ?? []).map((cur) => {
          const meta = getAuxTaskMetadata(cur.task);
          const isAuto = !cur || cur.provider === "auto" || !cur.provider;
          return (
            <div
              key={cur.task}
              data-testid="auxiliary-task-item"
              className="flex items-center justify-between gap-3 px-3 py-1.5 border border-border/30 bg-card/50 hover:bg-muted/60 hover:border-border/60 transition-colors"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-xs font-medium">{meta.label}</span>
                  <span className="text-xs text-text-tertiary">{meta.hint}</span>
                </div>
                <div className="text-xs font-mono text-text-secondary truncate">
                  {isAuto ? "auto (use main model)" : `${cur?.provider} · ${cur?.model || "(provider default)"}`}
                </div>
              </div>
              <Button
                size="sm"
                outlined
                onClick={() => setPicker({ kind: "aux", task: cur.task })}
                className="h-6 text-xs uppercase"
              >
                Change
              </Button>
            </div>
          );
        })}
      </div>
      {picker && picker.kind === "aux" && (
        <ModelPickerDialog
          key={`picker-${refreshKey}`}
          loader={api.getModelOptions}
          alwaysGlobal
          title={`Set Auxiliary: ${getAuxTaskMetadata(picker.task).label}`}
          onApply={async ({ provider, model, confirmExpensiveModel }) => {
            const result = await api.setModelAssignment({
              confirm_expensive_model: confirmExpensiveModel,
              scope: "auxiliary",
              task: picker.task,
              provider,
              model,
            });
            if (!result.confirm_required) onSaved();
            return result;
          }}
          onClose={() => setPicker(null)}
        />
      )}
    </>
  );
}

/* ─── Page ─── */

export default function ModelsPage() {
  const [pendingReloadModel, setPendingReloadModel] = useState<string | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const defaultTab = getTabFromQuery(searchParams);
  const [activeTab, setActiveTab] = useState(defaultTab);

  // Sync active tab with URL query param changes (resets to default when param absent/invalid)
  useEffect(() => {
    setActiveTab(getTabFromQuery(searchParams));
  }, [searchParams]);

  const [days, setDays] = useState(30);
  const [data, setData] = useState<ModelsAnalyticsResponse | null>(null);
  const [aux, setAux] = useState<AuxiliaryModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saveKey, setSaveKey] = useState(0);
  const [showTokens, setShowTokens] = useState(false);
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  // Settings panel state
  const [picker, setPicker] = useState<PickerTarget | null>(null);
  const [fallbacks, setFallbacks] = useState<FallbackEntry[]>([]);
  const fallbacksRef = useRef<FallbackEntry[]>([]);
  const [fallbackLoading, setFallbackLoading] = useState(false);
  const [fallbackBusy, setFallbackBusy] = useState(false);
  const fallbackBusyRef = useRef(false);
  const [fallbackError, setFallbackError] = useState<string | null>(null);
  const [pickerFallback, setPickerFallback] = useState<PickerTarget | null>(null);

  useEffect(() => {
    let cancelled = false;
    setFallbackLoading(true);
    api.getConfiguredModels()
      .then((cfg) => {
        if (cancelled) return;
        fallbacksRef.current = cfg.fallbacks;
        setFallbacks(cfg.fallbacks);
      })
      .catch((e) => {
        if (cancelled) return;
        setFallbackError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setFallbackLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    api.getConfig().then((cfg) => {
      const dash = (cfg?.dashboard ?? {}) as { show_token_analytics?: unknown };
      setShowTokens(dash.show_token_analytics === true);
    }).catch(() => { setShowTokens(false); });
  }, []);

  const mainProv = aux?.main.provider ?? "";
  const mainModel = aux?.main.model ?? "";



  const saveFallbackChain = async (next: FallbackEntry[], prev: FallbackEntry[]) => {
    if (fallbackBusyRef.current) return false;
    fallbackBusyRef.current = true;
    fallbacksRef.current = next;
    setFallbacks(next);
    setFallbackBusy(true);
    setFallbackError(null);
    try {
      const saved = await api.setFallbackChain(next);
      fallbacksRef.current = saved.fallbacks;
      setFallbacks(saved.fallbacks);
      setSaveKey((k) => k + 1);
      return true;
    } catch (e) {
      fallbacksRef.current = prev;
      setFallbacks(prev);
      setFallbackError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      fallbackBusyRef.current = false;
      setFallbackBusy(false);
    }
  };

  const moveFallback = async (from: number, to: number) => {
    if (fallbackBusyRef.current) return;
    const prev = fallbacksRef.current;
    const next = [...prev];
    const [item] = next.splice(from, 1);
    if (!item) return;
    next.splice(to, 0, item);
    await saveFallbackChain(next, prev);
  };

  const addFallback = async ({ provider, model }: { provider: string; model: string }) => {
    if (fallbackBusyRef.current) return;
    const prev = fallbacksRef.current;
    const next = [...prev, { provider, model }];
    const saved = await saveFallbackChain(next, prev);
    if (saved) setPickerFallback(null);
  };

  const removeFallback = async (idx: number) => {
    if (fallbackBusyRef.current) return;
    const prev = fallbacksRef.current;
    const next = prev.filter((_, i) => i !== idx);
    await saveFallbackChain(next, prev);
  };

  const load = useCallback(() => {
    setLoading(true); setError(null);
    Promise.all([api.getModelsAnalytics(days), api.getAuxiliaryModels().catch(() => null)])
      .then(([models, auxData]) => { setData(models); setAux(auxData); })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [days]);

  const refreshAux = useCallback(() => {
    api
      .getAuxiliaryModels()
      .then(setAux)
      .catch(() => {});
  }, []);

  const onAssigned = useCallback(() => {
    // Reload aux state after any assignment change.
    refreshAux();
    setSaveKey((k) => k + 1);
  }, [refreshAux]);

  useLayoutEffect(() => {
    if (activeTab !== "used-models") {
      setAfterTitle(null);
      setEnd(null);
      return () => {
        setAfterTitle(null);
        setEnd(null);
      };
    }
    setAfterTitle(
      <div className="flex flex-wrap items-center gap-1.5">
        {PERIODS.map((p) => (
          <Button
            key={p.label}
            type="button"
            size="sm"
            outlined={days !== p.days}
            onClick={() => setDays(p.days)}
            className="uppercase"
          >
            {p.label}
          </Button>
        ))}
        <Button
          type="button"
          ghost
          size="icon"
          className="text-muted-foreground hover:text-foreground"
          onClick={load}
          disabled={loading}
          aria-label={t.common.refresh}
        >
          {loading ? <Spinner /> : <RefreshCw />}
        </Button>
      </div>,
    );
    setEnd(null);
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [activeTab, days, loading, load, setAfterTitle, setEnd, t.common.refresh]);

  useEffect(() => { load(); }, [load]);

  const selectedTab = activeTab;
  const switchTab = (tab: string) => {
    if (VALID_TABS.has(tab)) {
      const nextParams = new URLSearchParams(searchParams);
      nextParams.set("tab", tab);
      setSearchParams(nextParams, { replace: true });
      setActiveTab(tab);
    }
  };

  // Model assignments can change outside this page (config editor, chat
  // /model --global, CLI), so refetch them when the page regains focus.
  useEffect(() => {
    let last = 0;
    const onFocus = () => {
      if (document.visibilityState !== "visible") return;
      if (Date.now() - last < 1000) return;
      last = Date.now();
      refreshAux();
    };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onFocus);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onFocus);
    };
  }, [refreshAux]);

  return (
    <div className="flex flex-col gap-6">
      <PluginSlot name="models:top" />

      {/* Tabbed content */}
      <div className="flex flex-col gap-4" data-testid="models-tabs">
            <TabsList className="mb-2">
              <TabsTrigger value="main-model" active={selectedTab === "main-model"} onClick={() => switchTab("main-model")} data-testid="models-settings-main-tab">Main Model</TabsTrigger>
              <TabsTrigger value="auxiliary-tasks" active={selectedTab === "auxiliary-tasks"} onClick={() => switchTab("auxiliary-tasks")} data-testid="models-settings-aux-tab">Auxiliary Tasks</TabsTrigger>
              <TabsTrigger value="used-models" active={selectedTab === "used-models"} onClick={() => switchTab("used-models")} data-testid="models-settings-used-tab">Used Models</TabsTrigger>
            </TabsList>

            {/* ── Main Model ── */}
            {selectedTab === "main-model" && (
              <div className="space-y-6" data-testid="settings-tab-panel">
                <Card data-testid="main-model-card">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between gap-3 flex-wrap">
                      <div className="flex items-center gap-2">
                        <Settings2 className="h-4 w-4 text-muted-foreground" />
                        <CardTitle className="text-sm">Main Model</CardTitle>
                        <span className="text-[10px] text-muted-foreground">primary model for new sessions</span>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3 pt-3">
                    <div className="flex items-center justify-between gap-3 bg-muted/20 border border-border/50 px-3 py-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 mb-0.5">
                          <Star className="h-3 w-3 text-primary" />
                          <span className="text-xs font-medium uppercase tracking-wider">Main model</span>
                        </div>
                        <div className="text-xs font-mono text-muted-foreground truncate">
                          {mainProv || "(unset)"}{mainProv && mainModel && " · "}{mainModel || "(unset)"}
                        </div>
                      </div>
                      <Button size="sm" onClick={() => setPicker({ kind: "main" })} className="text-xs">Change</Button>
                    </div>
                    {picker && picker.kind === "main" && (
                      <ModelPickerDialog
                        key={`picker-${saveKey}`}
                        loader={api.getModelOptions}
                        alwaysGlobal
                        title="Set Main Model"
                        onApply={async ({ provider, model, confirmExpensiveModel }) => {
                          const result = await api.setModelAssignment({
                            confirm_expensive_model: confirmExpensiveModel,
                            scope: "main",
                            task: "",
                            provider,
                            model,
                          });
                          if (!result.confirm_required) {
                            setPendingReloadModel(model.split("/").slice(-1)[0]);
                            onAssigned();
                          }
                          return result;
                        }}
                        onClose={() => setPicker(null)}
                      />
                    )}
                  </CardContent>
                </Card>

                {/* ── Fallback Chain (inside Main Model tab) ── */}
                <Card data-testid="fallback-chain-card">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between gap-3 flex-wrap">
                      <div className="flex items-center gap-2">
                        <Settings2 className="h-4 w-4 text-muted-foreground" />
                        <CardTitle className="text-sm">Fallback Providers</CardTitle>
                        <span className="text-[10px] text-muted-foreground">additional providers used if the main model is unavailable</span>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3 pt-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <RefreshCw className="h-3 w-3 text-muted-foreground" />
                        <span className="text-xs font-medium uppercase tracking-wider">Fallback chain</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Button size="sm" outlined onClick={() => setPickerFallback({ kind: "fallback" })} disabled={fallbackBusy} className="text-xs" data-testid="fallback-add-button">Add</Button>
                      </div>
                    </div>
                    {fallbackLoading && <div className="flex items-center justify-center py-4"><Spinner className="text-xs text-muted-foreground" /></div>}
                    {!fallbackLoading && fallbacks.length === 0 && (
                      <div className="text-[10px] text-muted-foreground/60 italic py-2">No fallback providers configured. Add one to continue when the main model fails.</div>
                    )}
                    {!fallbackLoading && fallbacks.length > 0 && (
                      <div className="space-y-1">
                        {fallbacks.map((fb, idx) => (
                          <div key={fallbackEntryKey(fb)} className="flex items-center gap-2 bg-muted/30 border border-border/50 px-3 py-2 rounded" data-testid={`fallback-item-${idx}`}>
                            <span className="text-xs text-muted-foreground/50 w-6 font-mono">{idx + 1}</span>
                            <span className="text-xs font-mono flex-1 truncate">{fb.provider} · {fb.model}</span>
                            <div className="flex items-center gap-1">
                              <button type="button" disabled={fallbackBusy || idx === 0} onClick={() => idx > 0 && moveFallback(idx, idx - 1)} className="flex items-center gap-1 px-2 py-1 text-xs bg-muted hover:bg-muted/80 border border-border rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors" aria-label="Move up" data-testid={`fallback-move-up-${idx}`}>
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="18 15 12 9 6 15"/></svg>
                                <span className="hidden sm:inline">Up</span>
                              </button>
                              <button type="button" disabled={fallbackBusy || idx === fallbacks.length - 1} onClick={() => idx < fallbacks.length - 1 && moveFallback(idx, idx + 1)} className="flex items-center gap-1 px-2 py-1 text-xs bg-muted hover:bg-muted/80 border border-border rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors" aria-label="Move down" data-testid={`fallback-move-down-${idx}`}>
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
                                <span className="hidden sm:inline">Down</span>
                              </button>
                              <button type="button" disabled={fallbackBusy} onClick={() => removeFallback(idx)} className="flex items-center gap-1 px-2 py-1 text-xs bg-destructive/10 hover:bg-destructive/20 border border-destructive/20 text-destructive rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors" aria-label="Remove" data-testid={`fallback-remove-${idx}`}>
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                                <span className="hidden sm:inline">Remove</span>
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    {fallbackError && <div className="text-[10px] text-destructive" data-testid="fallback-error">{fallbackError}</div>}
                    {pickerFallback && (
                      <ModelPickerDialog
                        key={`picker-fallback-${saveKey}`} loader={api.getModelOptions} alwaysGlobal confirmLabel="Save" title="Add Fallback Provider"
                        onApply={async ({ provider, model }) => { await addFallback({ provider, model }); }}
                        onClose={() => setPickerFallback(null)}
                      />
                    )}
                  </CardContent>
                </Card>

              </div>
            )}

            {/* ── Auxiliary Tasks ── */}
            {selectedTab === "auxiliary-tasks" && (
              <div data-testid="auxiliary-tasks-tab-panel">
                <AuxiliaryTasksPanel aux={aux} refreshKey={saveKey} onSaved={onAssigned} />
              </div>
            )}

            {/* ── Used Models ── */}
            {selectedTab === "used-models" && (
              <div data-testid="used-models-tab-panel" className="contents">
                {data && (
                  <Card className="mb-4">
                    <CardContent className="py-4">
                      <Stats items={
                        showTokens
                          ? [
                              { label: t.models.modelsUsed, value: String(data.totals.distinct_models) },
                              { label: t.analytics.totalTokens, value: formatTokens(data.totals.total_input + data.totals.total_output) },
                              { label: t.analytics.input, value: formatTokens(data.totals.total_input) },
                              { label: t.analytics.output, value: formatTokens(data.totals.total_output) },
                              { label: t.models.estimatedCost, value: formatCost(data.totals.total_estimated_cost) },
                              { label: t.analytics.totalSessions, value: String(data.totals.total_sessions) },
                            ]
                          : [
                              { label: t.models.modelsUsed, value: String(data.totals.distinct_models) },
                              { label: t.analytics.totalSessions, value: String(data.totals.total_sessions) },
                            ]
                      } />
                    </CardContent>
                  </Card>
                )}
                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-1.5">
                    {PERIODS.map((p) => (
                      <Button key={p.label} type="button" size="sm" outlined={days !== p.days} onClick={() => setDays(p.days)} data-testid={`used-models-period-${p.days}`}>{p.label}</Button>
                    ))}
                  </div>
                  <Button type="button" size="sm" outlined onClick={load} disabled={loading} prefix={loading ? <Spinner /> : <RefreshCw />} data-testid="used-models-refresh-button">
                    {t.common.refresh}
                  </Button>
                </div>
                {loading && !data && <div className="flex items-center justify-center py-24"><Spinner className="text-2xl text-primary" /></div>}
                {error && <Card><CardContent className="py-6"><p className="text-sm text-destructive text-center">{error}</p></CardContent></Card>}
                {data && (
                  <>
                    {data.models.length > 0 ? (
                      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" data-testid="used-models-grid">
                        {data.models.map((m, i) => (
                          <ModelCard key={`${m.model}:${m.provider}`} entry={m} rank={i + 1} main={aux?.main ?? null} aux={aux?.tasks ?? []} onAssigned={onAssigned} showTokens={showTokens} />
                        ))}
                      </div>
                    ) : (
                      <Card data-testid="used-models-empty-state">
                        <CardContent className="py-12">
                          <div className="flex flex-col items-center text-muted-foreground">
                            <Cpu className="h-8 w-8 mb-3 opacity-40" />
                            <p className="text-sm font-medium">{t.models.noModelsData}</p>
                            <p className="text-xs mt-1 text-muted-foreground/60">{t.models.startSession}</p>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

      <ModelReloadConfirm
        model={pendingReloadModel}
        onCancel={() => setPendingReloadModel(null)}
      />
      <PluginSlot name="models:bottom" />
    </div>
  );
}
