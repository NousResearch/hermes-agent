import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Plus, Route as RouteIcon, Trash2, Wand2 } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { ModelPickerDialog } from "@/components/ModelPickerDialog";
import { api } from "@/lib/api";
import { getNestedValue } from "@/lib/nested";
import {
  ROUTING_KEYS,
  addFallbackRoute,
  hasMainFallbackSupport,
  hasSchemaKey,
  moveFallbackRoute,
  parseFallbackRoutes,
  removeFallbackRoute,
  serializeFallbackRoutes,
  showsRoutingBlock,
  updateFallbackRoute,
  type FallbackRoute,
} from "@/lib/model-routing";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";
import type { Translations } from "@/i18n/types";

type RoutingStrings = NonNullable<Translations["config"]["modelRouting"]>;

/** Which provider:model route the open picker should fill. */
type PickerTarget =
  | { kind: "subagent" }
  | { kind: "subagentFallback"; index: number }
  | { kind: "mainFallback"; index: number };

interface Props {
  config: Record<string, unknown>;
  schema: Record<string, unknown>;
  /** Writes one dotted config key into the page's in-memory config (persisted by Save). */
  onChange: (key: string, value: unknown) => void;
}

function Hint({ children }: { children: string }) {
  if (!children) return null;
  return <span className="text-xs text-text-secondary">{children}</span>;
}

function ToggleRow({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex flex-col gap-0.5">
        <Label className="text-sm">{label}</Label>
        <Hint>{hint}</Hint>
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  );
}

/**
 * Editor state for one fallback chain.
 *
 * The value written to config (`onChange`) is the serialized YAML list, which
 * drops all-blank rows so they never reach config.yaml. The *editor* rows,
 * however, must keep a blank row: that is exactly what 'Add route' appends, and
 * what the user then fills in. Deriving the editor rows from config on every
 * render (the previous behaviour) re-parsed the just-added blank row away — the
 * 'Add route' button silently did nothing. So the editor owns a draft and only
 * re-derives it when the incoming config value changed from a value we did NOT
 * write (form reset, import, an external edit).
 *
 * Returns `[rows, commit]`; `commit` updates the draft AND persists the
 * serialized value in one shot.
 */
function useFallbackChain(
  config: Record<string, unknown>,
  key: string,
  onChange: (key: string, value: unknown) => void,
): [FallbackRoute[], (rows: FallbackRoute[]) => void] {
  const configValue = getNestedValue(config, key);
  const [rows, setRows] = useState<FallbackRoute[]>(() =>
    parseFallbackRoutes(configValue),
  );
  // What we last wrote (or read) — compared against the live config value to
  // tell our own writes apart from an external change.
  const lastWritten = useRef(
    JSON.stringify(serializeFallbackRoutes(parseFallbackRoutes(configValue))),
  );

  useEffect(() => {
    const incoming = JSON.stringify(
      serializeFallbackRoutes(parseFallbackRoutes(configValue)),
    );
    if (incoming !== lastWritten.current) {
      lastWritten.current = incoming;
      setRows(parseFallbackRoutes(configValue));
    }
  }, [configValue]);

  const commit = useCallback(
    (next: FallbackRoute[]) => {
      setRows(next);
      const serialized = serializeFallbackRoutes(next);
      lastWritten.current = JSON.stringify(serialized);
      onChange(key, serialized);
    },
    [key, onChange],
  );

  return [rows, commit];
}

/** One provider:model row in a fallback chain, matching the config YAML shape. */
function RouteRow({
  route,
  index,
  m,
  canMoveUp,
  canMoveDown,
  onUpdate,
  onRemove,
  onMoveUp,
  onMoveDown,
  onPick,
}: {
  route: FallbackRoute;
  index: number;
  m: RoutingStrings;
  canMoveUp: boolean;
  canMoveDown: boolean;
  onUpdate: (index: number, patch: Partial<FallbackRoute>) => void;
  onRemove: (index: number) => void;
  onMoveUp: (index: number) => void;
  onMoveDown: (index: number) => void;
  onPick: (index: number) => void;
}) {
  const n = index + 1;
  return (
    <li className="flex flex-wrap items-end gap-2 border border-border p-2">
      <div className="grid gap-1">
        <Label className="text-xs text-muted-foreground">{m.providerLabel}</Label>
        <Input
          value={route.provider}
          aria-label={`${m.providerLabel} ${n}`}
          placeholder={m.providerPlaceholder}
          onChange={(e) => onUpdate(index, { provider: e.target.value })}
          className="w-44"
        />
      </div>
      <div className="grid min-w-[12rem] flex-1 gap-1">
        <Label className="text-xs text-muted-foreground">{m.modelLabel}</Label>
        <Input
          value={route.model}
          aria-label={`${m.modelLabel} ${n}`}
          placeholder={m.modelPlaceholder}
          onChange={(e) => onUpdate(index, { model: e.target.value })}
          className="font-mono text-xs"
        />
      </div>
      <Button
        ghost
        size="icon"
        aria-label={`${m.moveUp ?? "Move up"} — ${n}`}
        title={m.moveUp ?? "Move up"}
        disabled={!canMoveUp}
        onClick={() => onMoveUp(index)}
      >
        <ChevronUp className="h-4 w-4" />
      </Button>
      <Button
        ghost
        size="icon"
        aria-label={`${m.moveDown ?? "Move down"} — ${n}`}
        title={m.moveDown ?? "Move down"}
        disabled={!canMoveDown}
        onClick={() => onMoveDown(index)}
      >
        <ChevronDown className="h-4 w-4" />
      </Button>
      <Button
        ghost
        size="icon"
        aria-label={`${m.pickModel} — ${n}`}
        title={m.pickModel}
        onClick={() => onPick(index)}
      >
        <Wand2 className="h-4 w-4" />
      </Button>
      <Button
        ghost
        size="icon"
        aria-label={`${m.removeRoute} — ${n}`}
        title={m.removeRoute}
        onClick={() => onRemove(index)}
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    </li>
  );
}

/**
 * Models → Model routing block.
 *
 * Surfaces the subagent preferred route (`delegation.provider`/`delegation.model`),
 * the subagent fallback chain (`delegation.fallback_providers`), the main-agent
 * fallback (`fallback_model`) and the `delegation.hot_reload_model` toggle.
 *
 * Every control is schema-gated: it only renders when the served config schema
 * exposes the key (the same rule the config pages follow). All reads/writes go
 * through the generic config GET/save path — this block never invents an
 * endpoint. The host page owns saving (Config page: its Save button; Models
 * page: the Save button under this card).
 */
export function ModelRoutingCard({ config, schema, onChange }: Props) {
  const { t } = useI18n();
  const m: RoutingStrings =
    t.config.modelRouting ?? (en.config.modelRouting as RoutingStrings);
  const [picker, setPicker] = useState<PickerTarget | null>(null);

  // Editor drafts for the two fallback chains. Hooks run before the bail-out
  // below so they're called unconditionally.
  const [subagentFallbackRoutes, commitSubagentFallback] = useFallbackChain(
    config,
    ROUTING_KEYS.subagentFallback,
    onChange,
  );
  const [mainFallbackRoutes, commitMainFallback] = useFallbackChain(
    config,
    ROUTING_KEYS.mainFallback,
    onChange,
  );

  const providerKey = hasSchemaKey(schema, ROUTING_KEYS.subagentProvider)
    ? ROUTING_KEYS.subagentProvider
    : null;
  const modelKey = hasSchemaKey(schema, ROUTING_KEYS.subagentModel)
    ? ROUTING_KEYS.subagentModel
    : null;
  const showSubagent = !!providerKey || !!modelKey;
  const showSubagentFallback = hasSchemaKey(schema, ROUTING_KEYS.subagentFallback);
  const showMainFallback = hasMainFallbackSupport(schema);
  const showHotReload = hasSchemaKey(schema, ROUTING_KEYS.hotReload);

  if (!showsRoutingBlock(schema)) {
    return null;
  }

  const subagentRoute = {
    provider: providerKey ? String(getNestedValue(config, providerKey) ?? "") : "",
    model: modelKey ? String(getNestedValue(config, modelKey) ?? "") : "",
  };
  const hotReload = getNestedValue(config, ROUTING_KEYS.hotReload) === true;

  const applyPick = (target: PickerTarget, provider: string, model: string) => {
    if (target.kind === "subagent") {
      if (providerKey) onChange(providerKey, provider);
      if (modelKey) onChange(modelKey, model);
      return;
    }
    if (target.kind === "subagentFallback") {
      commitSubagentFallback(
        updateFallbackRoute(subagentFallbackRoutes, target.index, { provider, model }),
      );
      return;
    }
    commitMainFallback(
      updateFallbackRoute(mainFallbackRoutes, target.index, { provider, model }),
    );
  };

  return (
    <Card>
      <CardHeader className="py-3 px-4">
        <CardTitle className="flex items-center gap-2 text-sm">
          <RouteIcon className="h-4 w-4" />
          {m.title}
        </CardTitle>
        <Hint>{m.subtitle}</Hint>
      </CardHeader>
      <CardContent className="flex flex-col gap-5 px-4 pb-4">
        {showSubagent && (
          <section className="flex flex-col gap-2">
            <Label className="text-sm">{m.subagentTitle}</Label>
            <Hint>{m.subagentHint}</Hint>
            {subagentRoute.provider || subagentRoute.model ? (
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="secondary" className="font-mono text-xs">
                  {subagentRoute.provider || "—"} · {subagentRoute.model || "—"}
                </Badge>
                <Button
                  size="sm"
                  outlined
                  prefix={<Wand2 className="h-3.5 w-3.5" />}
                  onClick={() => setPicker({ kind: "subagent" })}
                >
                  {m.pickModel}
                </Button>
                <Button
                  ghost
                  size="sm"
                  onClick={() => {
                    if (providerKey) onChange(providerKey, "");
                    if (modelKey) onChange(modelKey, "");
                  }}
                >
                  {m.clearRoute}
                </Button>
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-text-tertiary">{m.subagentEmpty}</span>
                <Button
                  size="sm"
                  outlined
                  prefix={<Wand2 className="h-3.5 w-3.5" />}
                  onClick={() => setPicker({ kind: "subagent" })}
                >
                  {m.pickModel}
                </Button>
              </div>
            )}
          </section>
        )}

        {showSubagentFallback && (
          <section className="flex flex-col gap-2 border-t border-border pt-4">
            <Label className="text-sm">{m.subagentFallbackTitle}</Label>
            <Hint>{m.subagentFallbackHint}</Hint>
            {subagentFallbackRoutes.length === 0 ? (
              <p className="text-xs text-text-tertiary">{m.routeEmpty}</p>
            ) : (
              <ul className="flex flex-col gap-2">
                {subagentFallbackRoutes.map((route, index) => (
                  <RouteRow
                    key={index}
                    route={route}
                    index={index}
                    m={m}
                    canMoveUp={index > 0}
                    canMoveDown={index < subagentFallbackRoutes.length - 1}
                    onUpdate={(i, patch) =>
                      commitSubagentFallback(
                        updateFallbackRoute(subagentFallbackRoutes, i, patch),
                      )
                    }
                    onRemove={(i) =>
                      commitSubagentFallback(
                        removeFallbackRoute(subagentFallbackRoutes, i),
                      )
                    }
                    onMoveUp={(i) =>
                      commitSubagentFallback(
                        moveFallbackRoute(subagentFallbackRoutes, i, "up"),
                      )
                    }
                    onMoveDown={(i) =>
                      commitSubagentFallback(
                        moveFallbackRoute(subagentFallbackRoutes, i, "down"),
                      )
                    }
                    onPick={(i) => setPicker({ kind: "subagentFallback", index: i })}
                  />
                ))}
              </ul>
            )}
            <div>
              <Button
                size="sm"
                outlined
                prefix={<Plus className="h-3.5 w-3.5" />}
                onClick={() =>
                  commitSubagentFallback(addFallbackRoute(subagentFallbackRoutes))
                }
              >
                {m.addRoute}
              </Button>
            </div>
          </section>
        )}

        {showMainFallback && (
          <section className="flex flex-col gap-2 border-t border-border pt-4">
            <Label className="text-sm">{m.mainTitle}</Label>
            <Hint>{m.mainHint}</Hint>
            {mainFallbackRoutes.length === 0 ? (
              <p className="text-xs text-text-tertiary">{m.routeEmpty}</p>
            ) : (
              <ul className="flex flex-col gap-2">
                {mainFallbackRoutes.map((route, index) => (
                  <RouteRow
                    key={index}
                    route={route}
                    index={index}
                    m={m}
                    canMoveUp={index > 0}
                    canMoveDown={index < mainFallbackRoutes.length - 1}
                    onUpdate={(i, patch) =>
                      commitMainFallback(
                        updateFallbackRoute(mainFallbackRoutes, i, patch),
                      )
                    }
                    onRemove={(i) =>
                      commitMainFallback(
                        removeFallbackRoute(mainFallbackRoutes, i),
                      )
                    }
                    onMoveUp={(i) =>
                      commitMainFallback(
                        moveFallbackRoute(mainFallbackRoutes, i, "up"),
                      )
                    }
                    onMoveDown={(i) =>
                      commitMainFallback(
                        moveFallbackRoute(mainFallbackRoutes, i, "down"),
                      )
                    }
                    onPick={(i) => setPicker({ kind: "mainFallback", index: i })}
                  />
                ))}
              </ul>
            )}
            <div>
              <Button
                size="sm"
                outlined
                prefix={<Plus className="h-3.5 w-3.5" />}
                onClick={() =>
                  commitMainFallback(addFallbackRoute(mainFallbackRoutes))
                }
              >
                {m.addRoute}
              </Button>
            </div>
          </section>
        )}

        {showHotReload && (
          <section className="border-t border-border pt-4">
            <ToggleRow
              label={m.hotReload}
              hint={m.hotReloadHint}
              checked={hotReload}
              onChange={(value) => onChange(ROUTING_KEYS.hotReload, value)}
            />
          </section>
        )}
      </CardContent>

      {picker && (
        <ModelPickerDialog
          title={
            picker.kind === "subagent" ? m.subagentTitle : picker.kind === "subagentFallback" ? m.subagentFallbackTitle : m.mainTitle
          }
          alwaysGlobal
          loader={(options) => api.getModelOptions({ refresh: options?.refresh })}
          onApply={({ provider, model }) => {
            applyPick(picker, provider, model);
          }}
          onClose={() => setPicker(null)}
        />
      )}
    </Card>
  );
}
