import { Button } from "@nous-research/ui/ui/components/button";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import type { GatewayClient } from "@/lib/gatewayClient";
import { Check, Search, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cn, themedBody } from "@/lib/utils";
import { fuzzyRank } from "@/lib/fuzzy";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";

function interpolate(
  template: string,
  values: Record<string, string | number>,
): string {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

/**
 * Two-stage model picker modal.
 *
 * Mirrors ui-tui/src/components/modelPicker.tsx:
 *   Stage 1: pick provider (authenticated providers only)
 *   Stage 2: pick model within that provider
 *
 * Two invocation modes:
 *
 * 1. Chat-session mode (ChatSidebar) — pass `gw` + `sessionId`. The picker
 *    loads options via `model.options` JSON-RPC and applies the choice via
 *    `config.set`, so expensive-model confirmation can happen before switch.
 *
 * 2. Standalone mode (ModelsPage, Config settings) — pass a `loader` and
 *    `onApply`. The picker fetches options via the REST endpoint and calls
 *    `onApply(provider, model, persistGlobal)` instead of emitting a slash
 *    command.  This lets the Models page reuse the same UI without
 *    requiring an open chat PTY.
 */

interface ModelOptionProvider {
  name: string;
  slug: string;
  models?: string[];
  total_models?: number;
  is_current?: boolean;
  warning?: string;
}

interface ModelOptionsResponse {
  model?: string;
  provider?: string;
  providers?: ModelOptionProvider[];
}

interface ExpensiveModelConfirmResponse {
  confirm_message?: string;
  confirm_required?: boolean;
  warning?: string;
}

interface ConfigSetResponse extends ExpensiveModelConfirmResponse {
  value?: string;
}

interface PendingExpensiveConfirm {
  message: string;
  model: string;
  persistGlobal: boolean;
  provider: string;
}

interface Props {
  /** Chat-mode: when present, picker emits a slash command via onSubmit. */
  gw?: GatewayClient;
  sessionId?: string;
  onSubmit?(slashCommand: string): void;

  /** Standalone-mode: when present (and onSubmit absent), picker calls onApply. */
  loader?(): Promise<ModelOptionsResponse>;
  onApply?(args: {
    confirmExpensiveModel?: boolean;
    provider: string;
    model: string;
    persistGlobal: boolean;
  }):
    | Promise<ExpensiveModelConfirmResponse | void>
    | ExpensiveModelConfirmResponse
    | void;

  onClose(): void;
  title?: string;
  /** If true, hides "Persist globally" checkbox — always saves to config.yaml. */
  alwaysGlobal?: boolean;
}

export function ModelPickerDialog(props: Props) {
  const {
    gw,
    sessionId,
    onSubmit,
    loader,
    onApply,
    onClose,
    title,
    alwaysGlobal = false,
  } = props;
  const { t } = useI18n();
  const copy = t.modelsPage ?? en.modelsPage!;
  const dialogTitle = title ?? copy.switchModel;
  const standalone = !!loader && !!onApply;

  const [providers, setProviders] = useState<ModelOptionProvider[]>([]);
  const [currentModel, setCurrentModel] = useState("");
  const [currentProviderSlug, setCurrentProviderSlug] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSlug, setSelectedSlug] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [query, setQuery] = useState("");
  const [persistGlobal, setPersistGlobal] = useState(alwaysGlobal);
  const [applying, setApplying] = useState(false);
  const [pendingConfirm, setPendingConfirm] =
    useState<PendingExpensiveConfirm | null>(null);
  const closedRef = useRef(false);

  // Load providers + models on open.
  useEffect(() => {
    closedRef.current = false;

    const promise = standalone
      ? (loader as () => Promise<ModelOptionsResponse>)()
      : (gw as GatewayClient).request<ModelOptionsResponse>(
          "model.options",
          sessionId ? { session_id: sessionId } : {},
        );

    promise
      .then((r) => {
        if (closedRef.current) return;
        const next = r?.providers ?? [];
        setProviders(next);
        setCurrentModel(String(r?.model ?? ""));
        setCurrentProviderSlug(String(r?.provider ?? ""));
        setSelectedSlug(
          (next.find((p) => p.is_current) ?? next[0])?.slug ?? "",
        );
        setSelectedModel("");
        setLoading(false);
      })
      .catch((e) => {
        if (closedRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });

    return () => {
      closedRef.current = true;
    };
    // Deliberately omit props from deps — stable for the dialog's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Esc closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const selectedProvider = useMemo(
    () => providers.find((p) => p.slug === selectedSlug) ?? null,
    [providers, selectedSlug],
  );

  const models = useMemo(
    () => selectedProvider?.models ?? [],
    [selectedProvider],
  );

  const trimmedQuery = query.trim();

  // Fuzzy-ranked providers: match on name + slug + the provider's model ids so
  // typing a model name surfaces its provider (preserves the prior behaviour
  // where a model match also revealed its provider).
  const filteredProviders = useMemo(
    () =>
      fuzzyRank(
        providers,
        trimmedQuery,
        (p) => `${p.name} ${p.slug} ${(p.models ?? []).join(" ")}`,
      ).map((r) => r.item),
    [providers, trimmedQuery],
  );

  // Fuzzy-ranked models carrying the matched character positions so the model
  // list can highlight why each entry matched.
  const filteredModels = useMemo(
    () =>
      fuzzyRank(models, trimmedQuery, (m) => m).map((r) => ({
        model: r.item,
        positions: r.positions,
      })),
    [models, trimmedQuery],
  );

  const canConfirm = !!selectedProvider && !!selectedModel && !applying;

  const applySelection = async (
    confirmExpensiveModel = false,
    forced?: PendingExpensiveConfirm,
  ) => {
    const providerSlug = forced?.provider ?? selectedProvider?.slug ?? "";
    const model = forced?.model ?? selectedModel;
    const shouldPersistGlobal = forced?.persistGlobal ?? persistGlobal;

    if (!providerSlug || !model || applying) return;

    if (standalone && onApply) {
      setApplying(true);
      try {
        const result = await onApply({
          confirmExpensiveModel,
          provider: providerSlug,
          model,
          persistGlobal: shouldPersistGlobal,
        });
        if (result?.confirm_required) {
          setPendingConfirm({
            provider: providerSlug,
            model,
            persistGlobal: shouldPersistGlobal,
            message:
              result.confirm_message || result.warning || copy.expensiveDefault,
          });
          return;
        }
        onClose();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setApplying(false);
      }
    } else if (gw && sessionId) {
      setApplying(true);
      try {
        const global = shouldPersistGlobal ? " --global" : "";
        const result = await gw.request<ConfigSetResponse>("config.set", {
          confirm_expensive_model: confirmExpensiveModel,
          key: "model",
          session_id: sessionId,
          value: `${model} --provider ${providerSlug}${global}`,
        });
        if (result?.confirm_required) {
          setPendingConfirm({
            provider: providerSlug,
            model,
            persistGlobal: shouldPersistGlobal,
            message:
              result.confirm_message || result.warning || copy.expensiveDefault,
          });
          return;
        }
        onClose();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setApplying(false);
      }
    } else if (onSubmit) {
      const global = shouldPersistGlobal ? " --global" : "";
      onSubmit(`/model ${model} --provider ${providerSlug}${global}`);
      onClose();
    }
  };

  const confirm = () => {
    if (!canConfirm) return;
    void applySelection();
  };

  // Portal to document.body: the main dashboard column in App.tsx is
  // `relative z-2`, which creates a stacking context that traps fixed
  // descendants below the app sidebar (z-50). Without the portal this
  // modal's z-[100] is scoped to z-2 and the sidebar covers its left
  // edge — visible especially in the Large theme variants where the
  // larger root font widens the dialog into the sidebar's column. See
  // Toast.tsx for the same pattern.
  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-picker-title"
    >
      <div
        className={cn(
          themedBody,
          "relative w-full max-w-3xl max-h-[80vh] border border-border bg-card shadow-2xl flex flex-col",
        )}
      >
        <Button
          ghost
          size="icon"
          onClick={onClose}
          className="absolute end-2 top-2 text-muted-foreground hover:text-foreground"
          aria-label={copy.close}
        >
          <X />
        </Button>

        <header className="p-5 pb-3 border-b border-border">
          <h2
            id="model-picker-title"
            className="font-mondwest text-display text-base tracking-wider"
          >
            {dialogTitle}
          </h2>
          <p className="text-xs text-muted-foreground mt-1 font-mono">
            {interpolate(copy.currentModel, {
              model: currentModel || copy.unknown,
            })}
            {currentProviderSlug && ` · ${currentProviderSlug}`}
          </p>
        </header>

        <div className="px-5 pt-3 pb-2 border-b border-border">
          <div className="relative">
            <Search className="absolute start-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              autoFocus
              placeholder={copy.filterProvidersModels}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="ps-7 h-8 text-sm"
            />
          </div>
        </div>

        <div className="flex-1 min-h-0 grid grid-cols-[200px_1fr] overflow-hidden">
          <ProviderColumn
            loading={loading}
            error={error}
            providers={filteredProviders}
            total={providers.length}
            selectedSlug={selectedSlug}
            query={trimmedQuery}
            onSelect={(slug) => {
              setSelectedSlug(slug);
              setSelectedModel("");
            }}
          />

          <ModelColumn
            provider={selectedProvider}
            models={filteredModels}
            allModels={models}
            selectedModel={selectedModel}
            currentModel={currentModel}
            currentProviderSlug={currentProviderSlug}
            onSelect={setSelectedModel}
            onConfirm={(m) => {
              setSelectedModel(m);
              void applySelection(false, {
                provider: selectedProvider?.slug ?? "",
                model: m,
                persistGlobal,
                message: "",
              });
            }}
          />
        </div>

        <footer className="border-t border-border p-3 flex items-center justify-between gap-3 flex-wrap">
          {alwaysGlobal ? (
            <span className="text-xs text-muted-foreground">
              {copy.savesToConfig}
            </span>
          ) : (
            <div className="flex items-center gap-2">
              <Checkbox
                checked={persistGlobal}
                id="model-picker-persist-global"
                onCheckedChange={(checked) =>
                  setPersistGlobal(checked === true)
                }
              />

              <Label
                className="font-mondwest normal-case tracking-normal text-xs text-muted-foreground cursor-pointer"
                htmlFor="model-picker-persist-global"
              >
                {copy.persistGlobally}
              </Label>
            </div>
          )}

          <div className="flex items-center gap-2 ms-auto">
            <Button outlined onClick={onClose} disabled={applying}>
              {copy.cancel}
            </Button>
            <Button onClick={confirm} disabled={!canConfirm}>
              {applying ? <Spinner /> : copy.switch}
            </Button>
          </div>
        </footer>
      </div>
      <ConfirmDialog
        open={!!pendingConfirm}
        title={copy.expensiveWarning}
        description={pendingConfirm?.message}
        destructive
        confirmLabel={copy.switchAnyway}
        cancelLabel={copy.cancel}
        loading={applying}
        onCancel={() => setPendingConfirm(null)}
        onConfirm={() => {
          const pending = pendingConfirm;
          if (!pending) return;
          setPendingConfirm(null);
          void applySelection(true, pending);
        }}
      />
    </div>,
    document.body,
  );
}

/* ------------------------------------------------------------------ */
/*  Provider column                                                    */
/* ------------------------------------------------------------------ */

function ProviderColumn({
  loading,
  error,
  providers,
  total,
  selectedSlug,
  query,
  onSelect,
}: {
  loading: boolean;
  error: string | null;
  providers: ModelOptionProvider[];
  total: number;
  selectedSlug: string;
  query: string;
  onSelect(slug: string): void;
}) {
  const { t } = useI18n();
  const copy = t.modelsPage ?? en.modelsPage!;
  return (
    <div className="border-e border-border overflow-y-auto">
      {loading && (
        <div className="flex items-center gap-2 p-4 text-xs text-muted-foreground">
          <Spinner className="text-xs" /> {copy.loading}
        </div>
      )}

      {error && <div className="p-4 text-xs text-destructive">{error}</div>}

      {!loading && !error && providers.length === 0 && (
        <div className="p-4 text-xs text-muted-foreground italic">
          {query
            ? copy.noMatches
            : total === 0
              ? copy.noAuthenticatedProviders
              : copy.noMatches}
        </div>
      )}

      {providers.map((p) => {
        const active = p.slug === selectedSlug;
        return (
          <ListItem
            key={p.slug}
            active={active}
            onClick={() => onSelect(p.slug)}
            className={`items-start text-xs border-s-2 ${
              active ? "border-s-primary" : "border-s-transparent"
            }`}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="font-medium truncate">{p.name}</span>
                {p.is_current && <CurrentTag />}
              </div>
              <div className="text-xs text-text-secondary font-mono truncate">
                {p.slug} ·{" "}
                {interpolate(copy.modelsCount, {
                  count: p.total_models ?? p.models?.length ?? 0,
                })}
              </div>
            </div>
          </ListItem>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Model column                                                       */
/* ------------------------------------------------------------------ */

function ModelColumn({
  provider,
  models,
  allModels,
  selectedModel,
  currentModel,
  currentProviderSlug,
  onSelect,
  onConfirm,
}: {
  provider: ModelOptionProvider | null;
  models: { model: string; positions: number[] }[];
  allModels: string[];
  selectedModel: string;
  currentModel: string;
  currentProviderSlug: string;
  onSelect(model: string): void;
  onConfirm(model: string): void;
}) {
  const { t } = useI18n();
  const copy = t.modelsPage ?? en.modelsPage!;
  if (!provider) {
    return (
      <div className="overflow-y-auto">
        <div className="p-4 text-xs text-muted-foreground italic">
          {copy.pickProvider}
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-y-auto">
      {provider.warning && (
        <div className="p-3 text-xs text-destructive border-b border-border">
          {provider.warning}
        </div>
      )}

      {models.length === 0 ? (
        <div className="p-4 text-xs text-muted-foreground italic">
          {allModels.length ? copy.noModelsMatch : copy.noModelsProvider}
        </div>
      ) : (
        models.map(({ model: m, positions }) => {
          const active = m === selectedModel;
          const isCurrent =
            m === currentModel && provider.slug === currentProviderSlug;

          return (
            <ListItem
              key={m}
              active={active}
              onClick={() => onSelect(m)}
              onDoubleClick={() => onConfirm(m)}
              className="px-3 py-1.5 text-xs font-mono"
            >
              <Check
                className={`h-3 w-3 shrink-0 ${active ? "text-primary" : "text-transparent"}`}
              />
              <span className="flex-1 truncate">
                <HighlightedText text={m} positions={positions} />
              </span>
              {isCurrent && <CurrentTag />}
            </ListItem>
          );
        })
      )}
    </div>
  );
}

function CurrentTag() {
  const { t } = useI18n();
  const copy = t.modelsPage ?? en.modelsPage!;
  return (
    <span className="text-display text-xs tracking-wider text-primary shrink-0">
      {copy.current}
    </span>
  );
}

/**
 * Render `text` with the characters at `positions` emphasised, so users can
 * see which characters their fuzzy query matched. Positions are indices into
 * `text`; out-of-range indices are ignored.
 */
function HighlightedText({
  text,
  positions,
}: {
  text: string;
  positions: number[];
}) {
  if (!positions.length) {
    return <>{text}</>;
  }

  const hit = new Set(positions);

  return (
    <>
      {Array.from(text).map((ch, i) =>
        hit.has(i) ? (
          <mark
            key={i}
            className="bg-transparent text-primary font-semibold underline underline-offset-2"
          >
            {ch}
          </mark>
        ) : (
          <span key={i}>{ch}</span>
        ),
      )}
    </>
  );
}
