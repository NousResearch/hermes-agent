import { useCallback, useEffect, useLayoutEffect, useState } from "react";
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
  X,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  AuxiliaryModelsResponse,
  AuxiliaryTaskAssignment,
  MoaConfigResponse,
  MoaModelSlot,
  ModelsAnalyticsModelEntry,
  ModelsAnalyticsResponse,
} from "@/lib/api";
import { timeAgo, cn, themedBody } from "@/lib/utils";
import { formatTokenCount } from "@/lib/format";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";
import { ModelPickerDialog } from "@/components/ModelPickerDialog";
import { ModelReloadConfirm } from "@/components/ModelReloadConfirm";

const PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

// Stable task IDs must match _AUX_TASK_SLOTS in hermes_cli/web_server.py;
// user-facing labels and hints live under modelSettings.auxTasks in the catalog.
const AUX_TASKS: readonly { key: string }[] = [
  { key: "vision" },
  { key: "web_extract" },
  { key: "compression" },
  { key: "skills_hub" },
  { key: "approval" },
  { key: "mcp" },
  { key: "title_generation" },
  { key: "triage_specifier" },
  { key: "kanban_decomposer" },
  { key: "profile_describer" },
  { key: "curator" },
] as const;

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

/** Short model name: strip vendor prefix like "openrouter/" or "anthropic/". */
function shortModelName(model: string): string {
  const slashIdx = model.indexOf("/");
  if (slashIdx > 0) return model.slice(slashIdx + 1);
  return model;
}

/** Extract vendor prefix from a model string like "anthropic/claude-opus-4.7" → "anthropic". */
function modelVendor(model: string, fallback?: string): string {
  const slashIdx = model.indexOf("/");
  if (slashIdx > 0) return model.slice(0, slashIdx);
  return fallback || "";
}

function TokenBar({
  input,
  output,
  cacheRead,
  reasoning,
}: {
  input: number;
  output: number;
  cacheRead: number;
  reasoning: number;
}) {
  const { t } = useI18n();
  const total = input + output + cacheRead + reasoning;
  if (total === 0) return null;

  // Segments carry a CSS color value (hex or `var(--token)`) rather than
  // a Tailwind class so the input/output series can pick up the active
  // theme's `--series-*-token` vars — see `themes/types.ts`
  // `ThemeSeriesColors`. The /60–/70 fade on the bar is applied via
  // color-mix on the same value so themes don't need to ship two
  // separate hex literals.
  const segments: Array<{ color: string; label: string; value: number }> = [
    {
      value: cacheRead,
      color: "#60a5fa",
      label: t.modelSettings.tokenLegend.cacheRead,
    }, // tailwind blue-400
    {
      value: reasoning,
      color: "#c084fc",
      label: t.modelSettings.tokenLegend.reasoning,
    }, // tailwind purple-400
    {
      value: input,
      color: "var(--series-input-token)",
      label: t.modelSettings.tokenLegend.input,
    },
    {
      value: output,
      color: "var(--series-output-token)",
      label: t.modelSettings.tokenLegend.output,
    },
  ].filter((s) => s.value > 0);

  return (
    <div className="space-y-1.5">
      {/* Stacked bar — segments fill proportionally to their share of total */}
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
            {/* Stepped fill pattern overlay */}
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
}: {
  capabilities: ModelsAnalyticsModelEntry["capabilities"];
}) {
  const { t } = useI18n();
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
          <Wrench className="h-2.5 w-2.5" />{" "}
          {t.modelSettings.capabilities.tools}
        </span>
      )}
      {capabilities.supports_vision && (
        <span className="inline-flex items-center gap-1 bg-blue-500/10 px-1.5 py-0.5 text-xs font-medium text-blue-600 dark:text-blue-400">
          <Eye className="h-2.5 w-2.5" /> {t.modelSettings.capabilities.vision}
        </span>
      )}
      {capabilities.supports_reasoning && (
        <span className="inline-flex items-center gap-1 bg-purple-500/10 px-1.5 py-0.5 text-xs font-medium text-purple-600 dark:text-purple-400">
          <Brain className="h-2.5 w-2.5" />{" "}
          {t.modelSettings.capabilities.reasoning}
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

/* ──────────────────────────────────────────────────────────────────── */
/*  Per-card "Use as" menu                                              */
/* ──────────────────────────────────────────────────────────────────── */

function UseAsMenu({
  provider,
  model,
  isMain,
  mainAuxTask,
  onAssigned,
}: {
  provider: string;
  model: string;
  /** True when this card's model+provider match config.yaml's main slot. */
  isMain: boolean;
  /** If this model is assigned to a specific aux task, that task's key. */
  mainAuxTask: string | null;
  onAssigned(): void;
}) {
  const { t } = useI18n();
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
      setError(t.modelSettings.missingProviderModel);
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
            result.confirm_message || t.modelPicker.expensiveWarningFallback,
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

  // Close on outside click.
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
    <div className={cn("relative", open && "z-20")} data-use-as-menu>
      <Button
        size="sm"
        outlined
        onClick={() => setOpen((v) => !v)}
        disabled={busy}
        className="h-6 px-2 text-xs uppercase"
        prefix={busy ? <Spinner /> : null}
      >
        {t.modelSettings.useAs} <ChevronDown className="h-3 w-3" />
      </Button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 min-w-[220px] border border-border bg-card shadow-lg">
          <button
            type="button"
            onClick={() => assign("main", "")}
            disabled={busy}
            className="flex w-full items-center justify-between px-3 py-2 text-xs uppercase hover:bg-muted/50 disabled:opacity-40"
          >
            <span className="flex items-center gap-2">
              <Star className="h-3 w-3" />
              {t.modelSettings.mainModel}
            </span>
            {isMain && (
              <span className="text-display text-xs tracking-wider text-primary">
                {t.modelPicker.currentTag}
              </span>
            )}
          </button>

          <div className="border-t border-border/50 px-3 py-1.5 text-display text-xs tracking-wider text-text-tertiary">
            {t.modelSettings.auxiliaryTask}
          </div>

          <button
            type="button"
            onClick={() => assign("auxiliary", "")}
            disabled={busy}
            className="flex w-full items-center justify-between px-3 py-1.5 text-xs uppercase hover:bg-muted/50 disabled:opacity-40"
          >
            <span>{t.modelSettings.allAuxiliaryTasks}</span>
          </button>

          {AUX_TASKS.map((task) => (
            <button
              key={task.key}
              type="button"
              onClick={() => assign("auxiliary", task.key)}
              disabled={busy}
              className="flex w-full items-center justify-between px-3 py-1.5 text-xs uppercase hover:bg-muted/50 disabled:opacity-40"
            >
              <span>
                {t.modelSettings.auxTasks[task.key]?.label ?? task.key}
              </span>
              {mainAuxTask === task.key && (
                <span className="text-display text-xs tracking-wider text-primary">
                  {t.modelPicker.currentTag}
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
        title={t.modelPicker.expensiveWarningTitle}
        description={pendingConfirm?.message}
        destructive
        confirmLabel={t.modelPicker.switchAnyway}
        cancelLabel={t.common.cancel}
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

/* ──────────────────────────────────────────────────────────────────── */
/*  ModelCard                                                           */
/* ──────────────────────────────────────────────────────────────────── */

function ModelCard({
  entry,
  rank,
  main,
  aux,
  onAssigned,
  showTokens,
}: {
  entry: ModelsAnalyticsModelEntry;
  rank: number;
  main: { provider: string; model: string } | null;
  aux: AuxiliaryTaskAssignment[];
  onAssigned(): void;
  showTokens: boolean;
}) {
  const { t } = useI18n();
  const provider = entry.provider || modelVendor(entry.model);
  const totalTokens = entry.input_tokens + entry.output_tokens;
  const caps = entry.capabilities;

  const isMain =
    !!main && main.provider === provider && main.model === entry.model;

  // First aux task currently using this model (if any).
  const mainAuxTask =
    aux.find((a) => a.provider === provider && a.model === entry.model)?.task ??
    null;

  return (
    <Card
      className={cn("min-w-0 max-w-full", isMain && "ring-1 ring-primary/40")}
    >
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
                  <Star className="h-2.5 w-2.5" /> {t.modelSettings.mainBadge}
                </span>
              )}
              {mainAuxTask && (
                <span className="inline-flex items-center bg-purple-500/10 px-1.5 py-0.5 text-display text-xs font-medium tracking-wider text-purple-600 dark:text-purple-400">
                  {t.modelSettings.auxBadge.replace(
                    "{task}",
                    t.modelSettings.auxTasks[mainAuxTask]?.label ?? mainAuxTask,
                  )}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-1">
              {provider && (
                <Badge tone="secondary" className="text-xs">
                  {provider}
                </Badge>
              )}
              {caps.context_window && caps.context_window > 0 && (
                <span className="text-xs text-text-secondary">
                  {formatTokenCount(caps.context_window)}{" "}
                  {t.modelSettings.contextShort}
                </span>
              )}
              {caps.max_output_tokens && caps.max_output_tokens > 0 && (
                <span className="text-xs text-text-secondary">
                  {formatTokenCount(caps.max_output_tokens)}{" "}
                  {t.modelSettings.maxOutputShort}
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
            <TokenBar
              input={entry.input_tokens}
              output={entry.output_tokens}
              cacheRead={entry.cache_read_tokens}
              reasoning={entry.reasoning_tokens}
            />

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
            {showTokens && entry.estimated_cost > 0 && (
              <span className="flex items-center gap-0.5">
                <DollarSign className="h-2.5 w-2.5" />
                {formatCost(entry.estimated_cost)}
              </span>
            )}
            {showTokens && entry.tool_calls > 0 && (
              <span className="flex items-center gap-0.5">
                <Zap className="h-2.5 w-2.5" />
                {entry.tool_calls} {t.models.toolCalls}
              </span>
            )}
          </div>
          {entry.last_used_at > 0 && <span>{timeAgo(entry.last_used_at)}</span>}
        </div>

        <CapabilityBadges capabilities={entry.capabilities} />
      </CardContent>
    </Card>
  );
}

/* ──────────────────────────────────────────────────────────────────── */
/*  Model Settings panel (top of page)                                  */
/* ──────────────────────────────────────────────────────────────────── */

type PickerTarget = { kind: "main" } | { kind: "aux"; task: string };

type MoaPickerTarget =
  | { kind: "reference"; index: number }
  | { kind: "aggregator" };

function AuxiliaryTasksModal({
  aux,
  refreshKey,
  onSaved,
  onClose,
}: {
  aux: AuxiliaryModelsResponse | null;
  refreshKey: number;
  onSaved(): void;
  onClose(): void;
}) {
  const { t } = useI18n();
  const [picker, setPicker] = useState<PickerTarget | null>(null);
  const [resetBusy, setResetBusy] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const modalRef = useModalBehavior({ open: true, onClose });

  const resetAllAux = async () => {
    setConfirmReset(false);
    setResetBusy(true);
    try {
      await api.setModelAssignment({
        scope: "auxiliary",
        task: "__reset__",
        provider: "",
        model: "",
      });
      onSaved();
    } finally {
      setResetBusy(false);
    }
  };

  return (
    <div
      ref={modalRef}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="aux-modal-title"
    >
      <div
        className={cn(
          themedBody,
          "relative w-full max-w-2xl max-h-[80vh] border border-border bg-card shadow-2xl flex flex-col",
        )}
      >
        <Button
          ghost
          size="icon"
          onClick={onClose}
          className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
          aria-label={t.common.close}
        >
          <X />
        </Button>

        <header className="p-5 pb-3 border-b border-border">
          <div className="flex items-center justify-between gap-3 pr-8">
            <h2
              id="aux-modal-title"
              className="font-mondwest text-display text-base tracking-wider"
            >
              {t.modelSettings.auxiliaryTasks}
            </h2>
            <Button
              size="sm"
              outlined
              onClick={() => setConfirmReset(true)}
              disabled={resetBusy}
              className="h-6 text-xs uppercase"
              prefix={resetBusy ? <Spinner /> : null}
            >
              {t.modelSettings.resetAll}
            </Button>
          </div>
          <p className="text-xs text-text-secondary mt-2">
            {t.modelSettings.auxiliaryDescription}
          </p>
        </header>

        <div className="flex-1 overflow-y-auto p-5 space-y-1">
          {AUX_TASKS.map((task) => {
            const cur = aux?.tasks.find((a) => a.task === task.key);
            const isAuto = !cur || cur.provider === "auto" || !cur.provider;
            const copy = t.modelSettings.auxTasks[task.key];
            return (
              <div
                key={task.key}
                className="flex items-center justify-between gap-3 px-3 py-2 border border-border/30 bg-card/50 hover:bg-muted/20 transition-colors"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs font-medium">
                      {copy?.label ?? task.key}
                    </span>
                    <span className="text-xs text-text-tertiary">
                      {copy?.hint ?? ""}
                    </span>
                  </div>
                  <div className="text-xs font-mono text-text-secondary truncate">
                    {isAuto
                      ? t.modelSettings.autoUseMain
                      : `${cur?.provider} · ${cur?.model || `(${t.modelSettings.providerDefault})`}`}
                  </div>
                </div>
                <Button
                  size="sm"
                  outlined
                  onClick={() => setPicker({ kind: "aux", task: task.key })}
                  className="h-6 text-xs uppercase"
                >
                  {t.modelSettings.change}
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
            title={t.modelSettings.setAuxiliary.replace(
              "{task}",
              t.modelSettings.auxTasks[picker.task]?.label ?? picker.task,
            )}
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
        <ConfirmDialog
          open={confirmReset}
          onCancel={() => setConfirmReset(false)}
          onConfirm={() => void resetAllAux()}
          title={t.modelSettings.resetAuxModels}
          description={t.modelSettings.resetAuxDescription}
          destructive
          confirmLabel={t.modelSettings.resetAll}
          loading={resetBusy}
        />
      </div>
    </div>
  );
}

function MoaModelsModal({
  config,
  refreshKey,
  onClose,
  onSaved,
}: {
  config: MoaConfigResponse;
  refreshKey: number;
  onClose(): void;
  onSaved(next: MoaConfigResponse): void;
}) {
  const { t } = useI18n();
  const [draft, setDraft] = useState<MoaConfigResponse>(config);
  const [selected, setSelected] = useState(
    config.default_preset || Object.keys(config.presets)[0] || "default",
  );
  const [newName, setNewName] = useState("");
  const [picker, setPicker] = useState<MoaPickerTarget | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const presetNames = Object.keys(draft.presets || {});
  const preset = draft.presets[selected] || draft.presets[presetNames[0]];
  const slotLabel = (slot: MoaModelSlot) =>
    `${slot.provider || `(${t.modelSettings.slotProvider})`} · ${slot.model || `(${t.modelSettings.slotModel})`}`;

  const updateSelectedPreset = (
    updater: (
      preset: MoaConfigResponse["presets"][string],
    ) => MoaConfigResponse["presets"][string],
  ) => {
    setDraft((prev) => ({
      ...prev,
      presets: {
        ...prev.presets,
        [selected]: updater(prev.presets[selected]),
      },
    }));
  };

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const saved = await api.saveMoaModels(draft);
      onSaved(saved);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const addPreset = () => {
    const name = newName.trim();
    if (!name || draft.presets[name]) return;
    const seed = preset || {
      reference_models: draft.reference_models,
      aggregator: draft.aggregator,
      reference_temperature: draft.reference_temperature,
      aggregator_temperature: draft.aggregator_temperature,
      max_tokens: draft.max_tokens,
      enabled: draft.enabled,
    };
    setDraft((prev) => ({
      ...prev,
      default_preset: prev.default_preset || name,
      presets: {
        ...prev.presets,
        [name]: { ...seed, reference_models: [...seed.reference_models] },
      },
    }));
    setSelected(name);
    setNewName("");
  };

  const deletePreset = () => {
    if (presetNames.length <= 1) return;
    const remaining = presetNames.filter((name) => name !== selected);
    const nextSelected = remaining[0];
    setDraft((prev) => {
      const next = { ...prev.presets };
      delete next[selected];
      return {
        ...prev,
        presets: next,
        default_preset:
          prev.default_preset === selected ? nextSelected : prev.default_preset,
        active_preset:
          prev.active_preset === selected ? "" : prev.active_preset,
      };
    });
    setSelected(nextSelected);
  };

  if (!preset) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4">
      <Card className="max-h-[85vh] w-full max-w-2xl overflow-auto">
        <CardHeader>
          <CardTitle className="text-sm">
            {t.modelSettings.configureMoaPresets}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-text-secondary">
            {t.modelSettings.moaDescription}
          </p>

          <div className="flex flex-wrap items-center gap-2">
            <select
              className="border border-border bg-background px-2 py-1 text-xs"
              value={selected}
              onChange={(event) => setSelected(event.target.value)}
            >
              {presetNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <Button
              size="sm"
              outlined
              onClick={() =>
                setDraft((prev) => ({ ...prev, default_preset: selected }))
              }
            >
              {t.modelSettings.setDefault}
            </Button>
            <Button
              size="sm"
              ghost
              disabled={presetNames.length <= 1}
              onClick={deletePreset}
            >
              {t.common.delete}
            </Button>
            <input
              className="border border-border bg-background px-2 py-1 text-xs"
              placeholder={t.modelSettings.newPresetPlaceholder}
              value={newName}
              onChange={(event) => setNewName(event.target.value)}
            />
            <Button
              size="sm"
              outlined
              disabled={!newName.trim() || !!draft.presets[newName.trim()]}
              onClick={addPreset}
            >
              {t.modelSettings.addPreset}
            </Button>
          </div>

          <div className="text-xs text-text-secondary">
            {t.modelSettings.defaultLabel.replace(
              "{name}",
              draft.default_preset,
            )}
          </div>

          <div className="space-y-2">
            <div className="text-display text-xs font-medium tracking-wider">
              {t.modelSettings.referenceModels}
            </div>
            {preset.reference_models.map((slot, index) => (
              <div
                key={`${selected}-${slot.provider}-${slot.model}-${index}`}
                className="flex items-center gap-2 border border-border/50 bg-muted/20 px-3 py-2"
              >
                <div className="min-w-0 flex-1 truncate font-mono text-xs text-text-secondary">
                  {slotLabel(slot)}
                </div>
                <Button
                  size="sm"
                  outlined
                  onClick={() => setPicker({ kind: "reference", index })}
                >
                  {t.modelSettings.change}
                </Button>
                <Button
                  size="sm"
                  ghost
                  disabled={preset.reference_models.length <= 1}
                  onClick={() =>
                    updateSelectedPreset((prev) => ({
                      ...prev,
                      reference_models: prev.reference_models.filter(
                        (_, i) => i !== index,
                      ),
                    }))
                  }
                >
                  {t.modelSettings.remove}
                </Button>
              </div>
            ))}
            <Button
              size="sm"
              outlined
              onClick={() =>
                updateSelectedPreset((prev) => ({
                  ...prev,
                  reference_models: [...prev.reference_models, prev.aggregator],
                }))
              }
            >
              {t.modelSettings.addReferenceModel}
            </Button>
          </div>

          <div className="space-y-2">
            <div className="text-display text-xs font-medium tracking-wider">
              {t.modelSettings.aggregator}
            </div>
            <div className="flex items-center gap-2 border border-border/50 bg-muted/20 px-3 py-2">
              <div className="min-w-0 flex-1 truncate font-mono text-xs text-text-secondary">
                {slotLabel(preset.aggregator)}
              </div>
              <Button
                size="sm"
                outlined
                onClick={() => setPicker({ kind: "aggregator" })}
              >
                {t.modelSettings.change}
              </Button>
            </div>
          </div>

          {error && <div className="text-xs text-destructive">{error}</div>}
          <div className="flex justify-end gap-2 pt-2">
            <Button ghost onClick={onClose} disabled={busy}>
              {t.common.cancel}
            </Button>
            <Button onClick={save} disabled={busy}>
              {busy ? t.common.saving : t.common.save}
            </Button>
          </div>
        </CardContent>
      </Card>
      {picker && (
        <ModelPickerDialog
          key={`moa-picker-${refreshKey}-${selected}-${picker.kind}-${picker.kind === "reference" ? picker.index : "agg"}`}
          loader={api.getModelOptions}
          alwaysGlobal
          title={t.modelSettings.selectMoaModel}
          onApply={async ({ provider, model }) => {
            if ((provider || "").toLowerCase() === "moa") {
              setError(t.modelSettings.moaRecursiveError);
              return;
            }
            setError(null);
            updateSelectedPreset((prev) => {
              if (picker.kind === "aggregator")
                return { ...prev, aggregator: { provider, model } };
              return {
                ...prev,
                reference_models: prev.reference_models.map((slot, i) =>
                  i === picker.index ? { provider, model } : slot,
                ),
              };
            });
          }}
          onClose={() => setPicker(null)}
        />
      )}
    </div>
  );
}

function ModelSettingsPanel({
  aux,
  refreshKey,
  onSaved,
}: {
  aux: AuxiliaryModelsResponse | null;
  refreshKey: number;
  onSaved(): void;
}) {
  const { t } = useI18n();
  const [auxModalOpen, setAuxModalOpen] = useState(false);
  const [moaModalOpen, setMoaModalOpen] = useState(false);
  const [moa, setMoa] = useState<MoaConfigResponse | null>(null);
  const [picker, setPicker] = useState<PickerTarget | null>(null);
  const [pendingReloadModel, setPendingReloadModel] = useState<string | null>(
    null,
  );

  const mainProv = aux?.main.provider ?? "";
  const mainModel = aux?.main.model ?? "";

  useEffect(() => {
    api
      .getMoaModels()
      .then(setMoa)
      .catch(() => setMoa(null));
  }, [refreshKey]);

  const applyAssignment = async ({
    scope,
    task,
    provider,
    model,
    confirmExpensiveModel,
  }: {
    confirmExpensiveModel?: boolean;
    scope: "main" | "auxiliary";
    task: string;
    provider: string;
    model: string;
  }) => {
    const result = await api.setModelAssignment({
      confirm_expensive_model: confirmExpensiveModel,
      scope,
      task,
      provider,
      model,
    });
    if (!result.confirm_required) onSaved();
    return result;
  };

  // Count how many aux tasks have overrides
  const auxOverrideCount =
    aux?.tasks.filter((a) => a.provider && a.provider !== "auto").length ?? 0;

  return (
    <Card className="min-w-0 max-w-full overflow-hidden">
      <CardHeader className="min-w-0 pb-3">
        <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <Settings2 className="h-4 w-4 shrink-0 text-muted-foreground" />
          <CardTitle className="text-sm">
            {t.modelSettings.modelSettings}
          </CardTitle>
          <span className="max-w-full min-w-0 text-xs text-text-secondary [overflow-wrap:anywhere]">
            {t.modelSettings.appliesToNewSessions}
          </span>
        </div>
      </CardHeader>

      <CardContent className="min-w-0 space-y-3 pt-3">
        {/* Main row */}
        <div className="flex min-w-0 flex-col gap-2 bg-muted/20 border border-border/50 px-3 py-2 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-0.5">
              <Star className="h-3 w-3 text-primary" />
              <span className="text-display text-xs font-medium tracking-wider">
                {t.modelSettings.mainModel}
              </span>
            </div>
            <div className="text-xs font-mono text-text-secondary truncate">
              {mainProv || `(${t.common.none})`}
              {mainProv && mainModel && " · "}
              {mainModel || `(${t.common.none})`}
            </div>
          </div>
          <Button
            size="sm"
            onClick={() => setPicker({ kind: "main" })}
            className="shrink-0 self-start text-xs uppercase sm:self-center"
          >
            {t.modelSettings.change}
          </Button>
        </div>

        {/* Auxiliary tasks summary + open modal */}
        <div className="flex min-w-0 flex-col gap-2 bg-muted/20 border border-border/50 px-3 py-2 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-0.5">
              <Cpu className="h-3 w-3 text-text-tertiary" />
              <span className="text-display text-xs font-medium tracking-wider">
                {t.modelSettings.auxiliaryTasks}
              </span>
            </div>
            <div className="text-xs font-mono text-text-secondary truncate">
              {auxOverrideCount > 0
                ? t.modelSettings.overrideSummary
                    .replace("{overrides}", String(auxOverrideCount))
                    .replace(
                      "{automatic}",
                      String(AUX_TASKS.length - auxOverrideCount),
                    )
                : t.modelSettings.allAutoSummary.replace(
                    "{count}",
                    String(AUX_TASKS.length),
                  )}
            </div>
          </div>
          <Button
            size="sm"
            outlined
            onClick={() => setAuxModalOpen(true)}
            className="shrink-0 self-start text-xs uppercase sm:self-center"
          >
            {t.modelSettings.configure}
          </Button>
        </div>

        <div className="flex min-w-0 flex-col gap-2 bg-muted/20 border border-border/50 px-3 py-2 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 mb-0.5">
              <Brain className="h-3 w-3 text-text-tertiary" />
              <span className="text-display text-xs font-medium tracking-wider">
                {t.modelSettings.moaName}
              </span>
            </div>
            <div className="text-xs font-mono text-text-secondary truncate">
              {moa
                ? t.modelSettings.moaReferencesSummary
                    .replace("{count}", String(moa.reference_models.length))
                    .replace(
                      "{aggregator}",
                      `${moa.aggregator.provider}/${shortModelName(moa.aggregator.model)}`,
                    )
                : t.modelSettings.notLoaded}
            </div>
          </div>
          <Button
            size="sm"
            outlined
            onClick={() => setMoaModalOpen(true)}
            disabled={!moa}
            className="shrink-0 self-start text-xs uppercase sm:self-center"
          >
            {t.modelSettings.configure}
          </Button>
        </div>

        {picker && (
          <ModelPickerDialog
            key={`picker-${refreshKey}`}
            loader={api.getModelOptions}
            alwaysGlobal
            title={t.modelSettings.setMainModel}
            onApply={async ({ provider, model, confirmExpensiveModel }) => {
              const result = await applyAssignment({
                confirmExpensiveModel,
                scope: "main",
                task: "",
                provider,
                model,
              });
              if (!result.confirm_required) {
                setPendingReloadModel(model.split("/").slice(-1)[0]);
              }
              return result;
            }}
            onClose={() => setPicker(null)}
          />
        )}

        {auxModalOpen && (
          <AuxiliaryTasksModal
            aux={aux}
            refreshKey={refreshKey}
            onSaved={onSaved}
            onClose={() => setAuxModalOpen(false)}
          />
        )}

        <ModelReloadConfirm
          model={pendingReloadModel}
          onCancel={() => setPendingReloadModel(null)}
        />
        {moaModalOpen && moa && (
          <MoaModelsModal
            config={moa}
            refreshKey={refreshKey}
            onSaved={(next) => {
              setMoa(next);
              onSaved();
            }}
            onClose={() => setMoaModalOpen(false)}
          />
        )}
      </CardContent>
    </Card>
  );
}

/* ──────────────────────────────────────────────────────────────────── */
/*  Page                                                                */
/* ──────────────────────────────────────────────────────────────────── */

export default function ModelsPage() {
  const [days, setDays] = useState(30);
  const [data, setData] = useState<ModelsAnalyticsResponse | null>(null);
  const [aux, setAux] = useState<AuxiliaryModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saveKey, setSaveKey] = useState(0);
  // Gate the token/cost UI on `dashboard.show_token_analytics`.  See
  // hermes_cli/config.py for the rationale: the numbers exclude auxiliary
  // calls and retries, so they're misleading next to provider billing.
  const [showTokens, setShowTokens] = useState(false);
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  useEffect(() => {
    api
      .getConfig()
      .then((cfg) => {
        const dash = (cfg?.dashboard ?? {}) as {
          show_token_analytics?: unknown;
        };
        setShowTokens(dash.show_token_analytics === true);
      })
      .catch(() => {
        // Default to hidden on any failure — safer than showing wrong numbers.
        setShowTokens(false);
      });
  }, []);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.getModelsAnalytics(days),
      api.getAuxiliaryModels().catch(() => null),
    ])
      .then(([models, auxData]) => {
        setData(models);
        setAux(auxData);
      })
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
    // Period selector + refresh both live in afterTitle so the controls
    // sit immediately next to the page title instead of being pinned to
    // the far-right `end` slot. The active period is conveyed by the
    // filled (non-outlined) button — no redundant period badge.
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
  }, [days, loading, load, setAfterTitle, setEnd, t.common.refresh]);

  useEffect(() => {
    const timer = window.setTimeout(load, 0);
    return () => window.clearTimeout(timer);
  }, [load]);

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
    <div className="flex min-w-0 max-w-full flex-col gap-6">
      <PluginSlot name="models:top" />

      <div className="grid min-w-0 gap-6 lg:grid-cols-2">
        <ModelSettingsPanel
          aux={aux}
          refreshKey={saveKey}
          onSaved={onAssigned}
        />

        {data && (
          <Card className="min-w-0 max-w-full overflow-hidden">
            <CardContent className="min-w-0 py-6">
              <div className="min-w-0 max-w-full [&_div.grid]:grid-cols-[auto_minmax(0,1fr)_auto]">
                <Stats
                  className="min-w-0"
                  items={
                    showTokens
                      ? [
                          {
                            label: t.models.modelsUsed,
                            value: String(data.totals.distinct_models),
                          },
                          {
                            label: t.analytics.totalTokens,
                            value: formatTokens(
                              data.totals.total_input +
                                data.totals.total_output,
                            ),
                          },
                          {
                            label: t.analytics.input,
                            value: formatTokens(data.totals.total_input),
                          },
                          {
                            label: t.analytics.output,
                            value: formatTokens(data.totals.total_output),
                          },
                          {
                            label: t.models.estimatedCost,
                            value: formatCost(data.totals.total_estimated_cost),
                          },
                          {
                            label: t.analytics.totalSessions,
                            value: String(data.totals.total_sessions),
                          },
                        ]
                      : [
                          {
                            label: t.models.modelsUsed,
                            value: String(data.totals.distinct_models),
                          },
                          {
                            label: t.analytics.totalSessions,
                            value: String(data.totals.total_sessions),
                          },
                        ]
                  }
                />
              </div>
              {!showTokens && (
                <p className="mt-4 text-xs text-text-tertiary leading-relaxed">
                  {t.modelSettings.analyticsHiddenBeforeConfig}{" "}
                  <span className="font-mono">
                    dashboard.show_token_analytics
                  </span>{" "}
                  {t.modelSettings.analyticsHiddenInConfig}{" "}
                  <a href="/config" className="underline">
                    {t.app.nav.config}
                  </a>{" "}
                  {t.modelSettings.analyticsHiddenAfterConfig}
                </p>
              )}
            </CardContent>
          </Card>
        )}
      </div>

      {loading && !data && (
        <div className="flex items-center justify-center py-24">
          <Spinner className="text-2xl text-primary" />
        </div>
      )}

      {error && (
        <Card>
          <CardContent className="py-6">
            <p className="text-sm text-destructive text-center">{error}</p>
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          {data.models.length > 0 ? (
            <div className="grid min-w-0 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {data.models.map((m, i) => (
                <ModelCard
                  key={`${m.model}:${m.provider}`}
                  entry={m}
                  rank={i + 1}
                  main={aux?.main ?? null}
                  aux={aux?.tasks ?? []}
                  onAssigned={onAssigned}
                  showTokens={showTokens}
                />
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12">
                <div className="flex flex-col items-center text-muted-foreground">
                  <Cpu className="h-8 w-8 mb-3 opacity-40" />
                  <p className="text-sm font-medium">{t.models.noModelsData}</p>
                  <p className="text-xs mt-1 text-text-tertiary">
                    {t.models.startSession}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      <PluginSlot name="models:bottom" />
    </div>
  );
}
