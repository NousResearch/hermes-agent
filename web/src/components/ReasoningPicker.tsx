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
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import {
  filterEffortOptions,
  normalizeEffort,
  VALID_EFFORTS,
} from "@/lib/reasoning-effort";

interface ReasoningPickerProps {
  /** Current model string from config — re-reads the saved effort when it
   *  changes (a different model may have been selected). */
  currentModel: string;
  /** Profile whose config should be read/written. */
  profile?: string;
  /** Bumped after the model picker saves, to re-read config in lockstep. */
  refreshKey?: number;
  /** Provider-known reasoning-effort dial values for `currentModel`.
   *  undefined → full option list (unknown); [] → no reasoning dial. */
  reasoningLevels?: string[] | null;
  /** Called after a successful change so the sidebar can show an "apply on
   *  /new or reload" notice, matching the model-switch UX. */
  onChanged?: (effort: string) => void;
}

export function ReasoningPicker({
  currentModel,
  profile,
  refreshKey = 0,
  reasoningLevels,
  onChanged,
}: ReasoningPickerProps) {
  const [effort, setEffort] = useState("medium");
  const [loadedFor, setLoadedFor] = useState("");
  const [savingFor, setSavingFor] = useState("");
  const lastFetchKeyRef = useRef("");
  const loadGeneration = useRef(0);
  const saveGeneration = useRef(0);
  const fetchKey = `${profile ?? ""}:${currentModel}:${refreshKey}`;
  const loaded = loadedFor === fetchKey;
  const saving = savingFor === fetchKey;
  const displayEffort = loaded ? effort : "medium";

  useLayoutEffect(() => {
    loadGeneration.current += 1;
  }, [fetchKey]);

  useLayoutEffect(() => {
    saveGeneration.current += 1;
  }, [fetchKey]);

  const options = filterEffortOptions(
    reasoningLevels,
    loaded ? effort : undefined,
  );
  const hasDial = options.length > 0 || reasoningLevels === undefined || reasoningLevels === null;

  useEffect(() => {
    if (fetchKey === lastFetchKeyRef.current) return;
    lastFetchKeyRef.current = fetchKey;
    const requestId = loadGeneration.current;
    void api
      .getConfig(profile)
      .then((cfg) => {
        if (loadGeneration.current !== requestId) return;
        const agent = (cfg?.agent as Record<string, unknown> | undefined) ?? {};
        setEffort(normalizeEffort(agent.reasoning_effort));
        setLoadedFor(fetchKey);
      })
      .catch(() => {
        // Best-effort: show the neutral default rather than reusing another
        // profile/model's saved effort when this scoped read fails.
        if (loadGeneration.current === requestId) {
          setEffort("medium");
          setLoadedFor(fetchKey);
        }
      });
  }, [fetchKey, profile]);

  const onSelect = useCallback(
    (next: string) => {
      if (!VALID_EFFORTS.has(next) || next === effort) return;
      const prev = effort;
      const requestScope = fetchKey;
      const requestGeneration = ++saveGeneration.current;
      setEffort(next); // optimistic
      setSavingFor(requestScope);
      // Read-modify-write the whole config — the dashboard's single-key save
      // pattern — so we never clobber sibling keys. `saveConfig` PUTs the full
      // object the agent boots from.
      void api
        .getConfig(profile)
        .then((cfg) => {
          if (saveGeneration.current !== requestGeneration) return undefined;
          const base = (cfg ?? {}) as Record<string, unknown>;
          const agent =
            base.agent && typeof base.agent === "object"
              ? { ...(base.agent as Record<string, unknown>) }
              : {};
          agent.reasoning_effort = next;
          return api.saveConfig({ ...base, agent }, profile);
        })
        .then((saved) => {
          if (saved === undefined || saveGeneration.current !== requestGeneration) return;
          onChanged?.(next);
        })
        .catch(() => {
          if (saveGeneration.current === requestGeneration) {
            setEffort(prev); // revert on failure
          }
        })
        .finally(() => {
          if (saveGeneration.current === requestGeneration) {
            setSavingFor("");
          }
        });
    },
    [effort, fetchKey, onChanged, profile],
  );

  if (!hasDial) {
    // Model has no reasoning dial on this provider (e.g. MiMo/Nemotron via
    // OpenCode Go sends no reasoning params). Hide the picker entirely —
    // a visible-but-inert dial is worse than none.
    return null;
  }

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
        value={displayEffort}
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
