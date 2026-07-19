import { useEffect, useState } from "react";

import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";

import { api, type ModelOptionsResponse } from "@/lib/api";
import { useI18n } from "@/i18n";

const INHERIT_VALUE = "__inherit__";

interface DelegationValue {
  model: string;
  provider: string;
}

/**
 * Guided picker for the `delegation.model` + `delegation.provider` config keys.
 *
 * Replaces the two bare free-text inputs that the schema-driven AutoField would
 * otherwise render. Mirrors the Desktop equivalent in
 * `apps/desktop/src/app/settings/delegation-model-provider-field.tsx` and the
 * `FallbackModelsField` pattern: provider + model selects sourced from the
 * gateway's `/api/model/options` response, with free-text fallback for
 * custom-endpoint providers that have no probed model catalog.
 *
 * "Inherit from main agent" is a first-class dropdown option that writes `""`
 * to both keys — the documented default
 * (`hermes_cli/config.py:2297-2298`, resolver at
 * `tools/delegate_tool.py:3056-3170`).
 *
 * Selecting a provider clears the model so a new provider never pairs with the
 * old provider's model. Out-of-catalog persisted models stay selectable so an
 * existing valid pick renders instead of blank.
 *
 * Uses the Dashboard's existing fetch-then-setState pattern (no react-query)
 * to match the rest of `web/src/pages/`.
 */
export function DelegationModelProviderField({
  config,
  onChange,
}: {
  config: Record<string, unknown>;
  /**
   * Atomic write of both `delegation.model` and `delegation.provider` on every
   * change. Caller merges both keys into the parent config record in a
   * single update — never persist a half-pair.
   */
  onChange: (next: DelegationValue) => void;
}) {
  const { t } = useI18n();
  const c = t.config;

  const [modelOptions, setModelOptions] = useState<ModelOptionsResponse | null>(
    null,
  );
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;

    api
      .getModelOptions()
      .then((res) => {
        if (!cancelled) setModelOptions(res);
      })
      .catch(() => {
        if (!cancelled) setLoadFailed(true);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const providers = (modelOptions?.providers ?? []).filter((p) => p.slug);

  // Top-level `model` + `provider` are the currently configured MAIN agent's
  // pick (what the subagent inherits when the user picks "Inherit from main
  // agent").
  const inheritedProvider = modelOptions?.provider ?? "";
  const inheritedModel = modelOptions?.model ?? "";

  const delegationBlock =
    config.delegation && typeof config.delegation === "object"
      ? (config.delegation as Record<string, unknown>)
      : {};
  const delegationModel = String(delegationBlock.model ?? "");
  const delegationProvider = String(delegationBlock.provider ?? "");

  const isInherit = !delegationModel && !delegationProvider;

  const selectedRow = providers.find((p) => p.slug === delegationProvider);
  const catalog = selectedRow?.models ?? [];

  // Keep an out-of-catalog persisted model selectable.
  const modelItems =
    delegationModel && !catalog.includes(delegationModel)
      ? [delegationModel, ...catalog]
      : catalog;

  const providerHasEmptyCatalog =
    delegationProvider !== "" && catalog.length === 0;

  // Compute "what's being inherited" once for the helper line.
  const inheritedDisplay =
    inheritedProvider && inheritedModel
      ? `${inheritedProvider} / ${inheritedModel}`
      : inheritedModel || inheritedProvider || c.delegationInheritUnknown;

  // Offline degradation: gateway unreachable, no cached options. Fall back to
  // free-text inputs (never lock the page) — matches the Desktop behavior.
  if (loadFailed && !modelOptions) {
    return (
      <div className="grid gap-1.5">
        <p className="text-xs text-warning">{c.delegationLoadFailed}</p>
        <Input
          className="text-xs"
          onChange={(e) => onChange({ provider: e.target.value, model: "" })}
          placeholder={c.delegationProviderLabel}
          value={delegationProvider}
        />
        <Input
          className="text-xs"
          onChange={(e) =>
            onChange({ provider: delegationProvider, model: e.target.value })
          }
          placeholder={c.delegationCustomModelPlaceholder}
          value={delegationModel}
        />
      </div>
    );
  }

  return (
    <div className="grid gap-1.5">
      <Label className="text-sm">{c.delegationProviderLabel}</Label>
      <Select
        value={isInherit ? INHERIT_VALUE : delegationProvider}
        onValueChange={(next) => {
          if (next === INHERIT_VALUE) {
            onChange({ provider: "", model: "" });
          } else {
            // Switching providers clears the model — old provider's model
            // paired with the new provider would never resolve at the gateway.
            onChange({ provider: next, model: "" });
          }
        }}
      >
        <SelectOption value={INHERIT_VALUE}>
          {c.delegationInheritFromMain}
        </SelectOption>
        {providers.map((p) => (
          <SelectOption key={p.slug} value={p.slug}>
            {p.name}
          </SelectOption>
        ))}
      </Select>

      {isInherit ? (
        <p className="text-xs text-text-secondary">
          {c.delegationCurrentlyInheriting(inheritedDisplay)}
        </p>
      ) : providerHasEmptyCatalog ? (
        <>
          <Label className="text-sm">{c.delegationModelLabel}</Label>
          <Input
            className="text-xs"
            onChange={(e) =>
              onChange({ provider: delegationProvider, model: e.target.value })
            }
            placeholder={c.delegationCustomModelPlaceholder}
            value={delegationModel}
          />
        </>
      ) : (
        <>
          <Label className="text-sm">{c.delegationModelLabel}</Label>
          <Select
            value={delegationModel}
            onValueChange={(next) =>
              onChange({ provider: delegationProvider, model: next })
            }
          >
            {modelItems.map((model) => (
              <SelectOption key={model} value={model}>
                {model}
              </SelectOption>
            ))}
          </Select>
        </>
      )}
    </div>
  );
}

/**
 * Render the picker only on the `delegation.model` row — the `provider` row
 * is collapsed into the picker to avoid two controls for one logical field.
 */
export function isDelegationModelPickerKey(schemaKey: string): boolean {
  return (
    schemaKey === "delegation.model" || schemaKey === "delegation.provider"
  );
}
