/**
 * ReasoningPicker — sets the main model's reasoning effort from the dashboard
 * Chat sidebar, mirroring the desktop app's composer effort radio.
 *
 * The dashboard previously only showed a read-only "Reasoning" capability
 * badge (see ModelInfoCard) with no way to actually choose the effort level —
 * unlike the desktop app, which exposes a radio in its model menu. This closes
 * that parity gap.
 *
 * Storage: the effort persists to config.yaml at `agent.reasoning_effort`
 * (the same key the TUI's `/reasoning <level>` command and the desktop radio
 * write). We read the whole config and write it back — the established
 * single-key pattern on the dashboard (see ConfigPage) — so the value lands in
 * the config the agent boots a fresh chat from. As with the model picker, the
 * running chat session adopts the change on the next `/new` or page reload;
 * we surface that hint rather than forcing a reload here.
 *
 * Profile scoping: the sidebar passes the chat profile explicitly, so this
 * reads/writes the same config the chat PTY was launched from.
 */

import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Brain } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import {
  EFFORT_OPTIONS,
  normalizeEffort,
  VALID_EFFORTS,
} from "@/lib/reasoning-effort";
import type { ModelOptionsResult } from "@hermes/shared";

interface ReasoningPickerProps {
  /** Current model string from config — re-reads the saved effort when it
   *  changes (a different model may have been selected). */
  currentModel: string;
  /** Profile whose config should be read/written. */
  profile?: string;
  /** Bumped after the model picker saves, to re-read config in lockstep. */
  refreshKey?: number;
  /** Called after a successful change so the sidebar can show an "apply on
   *  /new or reload" notice, matching the model-switch UX. */
  onChanged?: (effort: string) => void;
}

/** A provider declaration is authoritative when present, including an empty
 * declaration. Keep a saved effort visible so changing models never silently
 * rewrites config before the user explicitly selects a new value. */
export function reasoningPickerOptions(effort: string, supportedEfforts: readonly string[] | undefined) {
  if (supportedEfforts === undefined) return EFFORT_OPTIONS;
  if (supportedEfforts.length === 0) return [];

  const supported = EFFORT_OPTIONS.filter(option => supportedEfforts.includes(option.value));
  return supported.some(option => option.value === effort) ? supported : [...supported, { label: effort, value: effort }];
}

export function ReasoningPicker({
  currentModel,
  profile,
  refreshKey = 0,
  onChanged,
}: ReasoningPickerProps) {
  const [effort, setEffort] = useState("medium");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const lastFetchKeyRef = useRef("");
  const fetchGenerationRef = useRef(0);
  const [supportedEfforts, setSupportedEfforts] = useState<string[] | undefined>();

  useEffect(() => {
    const fetchKey = `${profile ?? ""}:${currentModel}:${refreshKey}`;
    if (fetchKey === lastFetchKeyRef.current) return;
    lastFetchKeyRef.current = fetchKey;
    const generation = ++fetchGenerationRef.current;
    setLoaded(false);
    setSaving(false);
    setSupportedEfforts(undefined);

    void api
      .getConfig(profile)
      .then((cfg: Record<string, unknown>) => {
        if (generation !== fetchGenerationRef.current) return;
        const agent = (cfg?.agent as Record<string, unknown> | undefined) ?? {};
        setEffort(normalizeEffort(agent.reasoning_effort));
        setLoaded(true);
      })
      .catch(() => {
        if (generation !== fetchGenerationRef.current) return;
        // Config is authoritative for the displayed value; do not show stale
        // data from another profile when it cannot be read.
        setEffort("medium");
        setLoaded(true);
      });

    void api
      .getModelOptions({ profile })
      .then((options: ModelOptionsResult) => {
        if (generation !== fetchGenerationRef.current) return;
        const row = options.providers.find(provider => provider.is_current) ?? options.providers.find(provider => provider.models?.includes(currentModel));
        setSupportedEfforts(row?.capabilities?.[currentModel]?.reasoning_efforts ?? undefined);
      })
      .catch(() => {
        // Options are optional metadata. A same-scope failure leaves the
        // unrestricted fallback; a superseded response cannot alter it.
      });
  }, [currentModel, profile, refreshKey]);

  const onSelect = useCallback(
    (next: string) => {
      if (!VALID_EFFORTS.has(next) || next === effort) return;
      const prev = effort;
      const generation = fetchGenerationRef.current;
      setEffort(next); // optimistic
      setSaving(true);
      // Sparse patch: PUT /api/config deep-merges onto disk, so sending only
      // the edited key never clobbers sibling keys — and never echoes a
      // default-expanded snapshot back over values another surface changed
      // meanwhile (a CLI-pinned auxiliary slot would come back as "auto").
      void api
        .saveConfig({ agent: { reasoning_effort: next } }, profile)
        .then(() => {
          if (generation !== fetchGenerationRef.current) return;
          onChanged?.(next);
        })
        .catch(() => {
          if (generation !== fetchGenerationRef.current) return;
          setEffort(prev); // revert on failure
        })
        .finally(() => {
          if (generation === fetchGenerationRef.current) setSaving(false);
        });
    },
    [effort, onChanged, profile],
  );

  const options = reasoningPickerOptions(effort, supportedEfforts);
  if (supportedEfforts?.length === 0) return null;

  return (
    <div className="flex items-center gap-2 px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 text-text-tertiary">
        <Brain className="h-3.5 w-3.5" />
        <span className="text-display tracking-wider">reasoning</span>
      </div>
      <Select
        className="ml-auto min-w-0"
        disabled={!loaded || saving}
        onValueChange={onSelect}
        value={effort}
      >
        {options.map((opt) => (
          <SelectOption key={opt.value} value={opt.value}>
            {opt.label}
          </SelectOption>
        ))}
      </Select>
    </div>
  );
}
