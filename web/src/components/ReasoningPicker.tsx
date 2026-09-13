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

import { Button } from "@nous-research/ui/ui/components/button";
import { Brain, ChevronDown } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import {
  EFFORT_OPTIONS,
  normalizeEffort,
  VALID_EFFORTS,
} from "@/lib/reasoning-effort";
import { ReasoningPickerDialog } from "@/components/ReasoningPickerDialog";

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

export function ReasoningPicker({
  currentModel,
  profile,
  refreshKey = 0,
  onChanged,
}: ReasoningPickerProps) {
  const [effort, setEffort] = useState("medium");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const lastFetchKeyRef = useRef("");

  useEffect(() => {
    const fetchKey = `${profile ?? ""}:${currentModel}:${refreshKey}`;
    if (fetchKey === lastFetchKeyRef.current) return;
    lastFetchKeyRef.current = fetchKey;
    void api
      .getConfig(profile)
      .then((cfg) => {
        const agent = (cfg?.agent as Record<string, unknown> | undefined) ?? {};
        setEffort(normalizeEffort(agent.reasoning_effort));
        setLoaded(true);
      })
      .catch(() => {
        // Best-effort: keep the last known value rather than blanking it.
        setLoaded(true);
      });
  }, [currentModel, profile, refreshKey]);

  // Returns a promise so the dialog can await the save (success or the
  // revert-on-failure path) before dismissing itself, rather than closing
  // immediately and leaving any error surfacing entirely to the sidebar.
  const onSelect = useCallback(
    (next: string): Promise<void> => {
      if (!VALID_EFFORTS.has(next) || next === effort) return Promise.resolve();
      const prev = effort;
      setEffort(next); // optimistic
      setSaving(true);
      // Read-modify-write the whole config — the dashboard's single-key save
      // pattern — so we never clobber sibling keys. `saveConfig` PUTs the full
      // object the agent boots from.
      return api
        .getConfig(profile)
        .then((cfg) => {
          const base = (cfg ?? {}) as Record<string, unknown>;
          const agent =
            base.agent && typeof base.agent === "object"
              ? { ...(base.agent as Record<string, unknown>) }
              : {};
          agent.reasoning_effort = next;
          return api.saveConfig({ ...base, agent }, profile);
        })
        .then(() => {
          onChanged?.(next);
        })
        .catch(() => {
          setEffort(prev); // revert on failure
        })
        .finally(() => setSaving(false));
    },
    [effort, onChanged, profile],
  );

  const currentLabel =
    EFFORT_OPTIONS.find((o) => o.value === effort)?.label ?? effort;

  return (
    <div className="flex items-center gap-2 px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 text-text-tertiary">
        <Brain className="h-3.5 w-3.5" />
        <span className="text-display tracking-wider">reasoning</span>
      </div>
      <Button
        ghost
        size="sm"
        disabled={!loaded || saving}
        onClick={() => setPickerOpen(true)}
        className="ml-auto min-w-0 max-w-full px-0 py-0 normal-case tracking-normal text-xs font-medium hover:underline disabled:no-underline"
        title="change reasoning effort"
      >
        <span className="flex min-w-0 max-w-full items-center gap-1">
          <span className="truncate">{currentLabel}</span>
          <ChevronDown className="size-3.5 shrink-0 text-text-secondary" />
        </span>
      </Button>

      {pickerOpen && (
        <ReasoningPickerDialog
          currentEffort={effort}
          onSelect={onSelect}
          onClose={() => setPickerOpen(false)}
        />
      )}
    </div>
  );
}
