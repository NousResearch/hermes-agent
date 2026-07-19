import { useEffect, useRef, useState } from "react";

import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";

import { api, type ModelOptionsResponse } from "@/lib/api";
import { useI18n } from "@/i18n";

const INHERIT_VALUE = "__inherit__"

// Sentinel for the "model-only" configuration: `delegation.model` is set
// while `delegation.provider` is blank. The resolver at
// `tools/delegate_tool.py:3139-3149` preserves parent credentials in this
// state (documented in `website/docs/user-guide/configuration.md:2023`).
// Surfacing this as a distinct dropdown option lets users create and edit
// the state explicitly instead of getting it stripped by the picker.
const MODEL_ONLY_VALUE = "__model_only__";

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
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    // `isLoading` is initialized to `true`, so no explicit `setIsLoading(true)`
    // call here — keeps the effect purely about the async fetch and avoids
    // the `set-state-in-effect` lint rule.
    api
      .getModelOptions()
      .then((res) => {
        if (!cancelled) setModelOptions(res);
      })
      .catch(() => {
        if (!cancelled) setLoadFailed(true);
      })
      .finally(() => {
        // Gate the loading flag on `cancelled` so an unmount-then-remount
        // can't show a "not loading, but no options" flicker.
        if (!cancelled) setIsLoading(false);
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
  // Snapshot the persisted values once on mount — used to seed the local
  // draft state and to detect external config changes (profile switch,
  // reset-to-defaults). Same pattern as `FallbackModelsField` on the
  // Desktop side.
  const persistedModel = String(delegationBlock.model ?? "");
  const persistedProvider = String(delegationBlock.provider ?? "");

  // Local draft state — owned by the component so mid-edit values
  // (e.g. an empty input while typing) don't flip the persisted state
  // derivation and snap the dropdown back to "Inherit". The parent only
  // sees the commit emitted via `onChange`.
  const [draftProvider, setDraftProvider] = useState(persistedProvider);
  const [draftModel, setDraftModel] = useState(persistedModel);

  // The dropdown's selection is user INTENT — distinct from the
  // persisted model/provider pair. Tracking it separately means selecting
  // "Custom model" from a clean inherit state immediately shows the
  // model-only branch (with an empty input ready for typing), instead of
  // snapping back to inherit because the model is empty. Same for
  // backspace mid-edit: the dropdown stays where the user left it.
  const initialSelectValue = !persistedProvider && !persistedModel
    ? INHERIT_VALUE
    : !persistedProvider && persistedModel
      ? MODEL_ONLY_VALUE
      : persistedProvider;
  const [providerSelectValue, setProviderSelectValue] = useState<string>(initialSelectValue);

  // Track the last pair we emitted so we can ignore the autosave echo
  // through `config` (parent writes through `onChange`, then the new
  // config round-trips back through the `config` prop).
  const lastEmittedRef = useRef({ provider: persistedProvider, model: persistedModel });

  // Resync local draft when the persisted config changes from OUTSIDE
  // (profile switch, reset). Skip when the change is just our own emit
  // echoing back through the parent.
  useEffect(() => {
    const emitted = lastEmittedRef.current
    if (persistedProvider === emitted.provider && persistedModel === emitted.model) {
      return
    }
    setDraftProvider(persistedProvider)
    setDraftModel(persistedModel)
    // Also resync the dropdown's selected value to match the new
    // persisted state — otherwise a profile switch with the same model
    // but different provider would leave the dropdown showing the old
    // selection.
    setProviderSelectValue(
      !persistedProvider && !persistedModel
        ? INHERIT_VALUE
        : !persistedProvider && persistedModel
          ? MODEL_ONLY_VALUE
          : persistedProvider
    )
    lastEmittedRef.current = { provider: persistedProvider, model: persistedModel }
  }, [persistedProvider, persistedModel])

  const commit = (next: { provider: string; model: string }) => {
    lastEmittedRef.current = next
    onChange(next)
  }

  // Three distinct picker states — derived from the LOCAL DRAFT, not
  // the persisted config, so the model field can be empty in the
  // model-only state without the picker flipping back to inherit.
  //  - `isInherit`: both fields blank → use the parent's main model + credentials.
  //  - `isModelOnly`: dropdown is on MODEL_ONLY_VALUE → use the explicit
  //    model but inherit credentials from the parent (preserved by the
  //    resolver at `tools/delegate_tool.py:3139-3149`).
  //  - explicit pick → fully pinned to the chosen provider.
  const isInherit =
    providerSelectValue === INHERIT_VALUE ||
    (draftProvider === "" && draftModel === "");
  const isModelOnly =
    providerSelectValue === MODEL_ONLY_VALUE && draftProvider === "";

  const selectedRow = providers.find((p) => p.slug === draftProvider)
  const catalog = selectedRow?.models ?? []

  // Keep an out-of-catalog persisted model selectable.
  const modelItems =
    draftModel && !catalog.includes(draftModel)
      ? [draftModel, ...catalog]
      : catalog

  const providerHasEmptyCatalog =
    draftProvider !== "" && catalog.length === 0

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
          onChange={(e) => {
            setDraftProvider(e.target.value)
            commit({ provider: e.target.value, model: draftModel })
          }}
          placeholder={c.delegationProviderLabel}
          value={draftProvider}
        />
        <Input
          className="text-xs"
          onChange={(e) => {
            setDraftModel(e.target.value)
            commit({ provider: draftProvider, model: e.target.value })
          }}
          placeholder={c.delegationCustomModelPlaceholder}
          value={draftModel}
        />
      </div>
    )
  }

  return (
    <div className="grid gap-1.5">
      <Label className="text-sm">{c.delegationProviderLabel}</Label>
      <Select
        disabled={isLoading}
        value={providerSelectValue}
        onValueChange={(next) => {
          // The dropdown's selected value is user INTENT — track it
          // explicitly so it doesn't flip back when the persisted model
          // is briefly empty (mid-edit, model-only entry, etc).
          setProviderSelectValue(next)
          if (next === INHERIT_VALUE) {
            // Full inherit — clear both fields locally and commit.
            setDraftProvider("")
            setDraftModel("")
            commit({ provider: "", model: "" })
          } else if (next === MODEL_ONLY_VALUE) {
            // Model-only — clear the provider, keep the current draft
            // model (preserves whatever the user typed mid-edit) so the
            // resolver sees a coherent `provider="", model="…"` pair.
            setDraftProvider("")
            commit({ provider: "", model: draftModel })
          } else {
            // Switching to an explicit provider clears the model — the old
            // provider's model wouldn't resolve at the gateway.
            setDraftProvider(next)
            setDraftModel("")
            commit({ provider: next, model: "" })
          }
        }}
      >
        <SelectOption value={INHERIT_VALUE}>
          {c.delegationInheritFromMain}
        </SelectOption>
        <SelectOption value={MODEL_ONLY_VALUE}>
          {c.delegationModelOnlyOption}
        </SelectOption>
        {providers.map((p) => (
          <SelectOption key={p.slug} value={p.slug}>
            {p.name}
          </SelectOption>
        ))}
      </Select>
      {isLoading ? (
        <p aria-live="polite" className="text-xs text-text-secondary">
          {c.delegationLoading}
        </p>
      ) : null}

      {isInherit ? (
        <p className="text-xs text-text-secondary">
          {c.delegationCurrentlyInheriting(inheritedDisplay)}
        </p>
      ) : isModelOnly ? (
        // Model-only/provider-inherit — free-text input so the user can
        // edit the model; helper line explains the credentials-inherited
        // behavior. Stays mounted across edits so a brief empty value
        // (mid-type) doesn't unmount the input.
        <>
          <Label className="text-sm">{c.delegationModelLabel}</Label>
          <Input
            aria-label={c.delegationModelLabel}
            className="text-xs"
            onChange={(e) => {
              setDraftModel(e.target.value)
              commit({ provider: "", model: e.target.value })
            }}
            placeholder={c.delegationCustomModelPlaceholder}
            value={draftModel}
          />
          <p className="text-xs text-text-secondary">
            {c.delegationCredentialsInherited}
          </p>
        </>
      ) : providerHasEmptyCatalog ? (
        <>
          <Label className="text-sm">{c.delegationModelLabel}</Label>
          <Input
            aria-label={c.delegationModelLabel}
            className="text-xs"
            onChange={(e) => {
              setDraftModel(e.target.value)
              commit({ provider: draftProvider, model: e.target.value })
            }}
            placeholder={c.delegationCustomModelPlaceholder}
            value={draftModel}
          />
        </>
      ) : (
        <>
          <Label className="text-sm">{c.delegationModelLabel}</Label>
          <Select
            value={draftModel}
            onValueChange={(next) => {
              setDraftModel(next)
              commit({ provider: draftProvider, model: next })
            }}
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
