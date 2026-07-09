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

import { useI18n } from "@/i18n";
import { api } from "@/lib/api";
import {
  EFFORT_OPTIONS,
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
  const { t } = useI18n();
  const [effort, setEffort] = useState("medium");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
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

  const onSelect = useCallback(
    (next: string) => {
      if (!VALID_EFFORTS.has(next) || next === effort) return;
      const prev = effort;
      setEffort(next); // optimistic
      setSaving(true);
      // Read-modify-write the whole config — the dashboard's single-key save
      // pattern — so we never clobber sibling keys. `saveConfig` PUTs the full
      // object the agent boots from.
      void api
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

  return (
    <div className="flex items-center gap-2 px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 text-text-tertiary">
        <Brain className="h-3.5 w-3.5" />
        <span className="text-display tracking-wider">
          {t.chatSidebar.reasoning}
        </span>
      </div>
      <Select
        className="ml-auto min-w-0"
        disabled={!loaded || saving}
        onValueChange={onSelect}
        value={effort}
      >
        {EFFORT_OPTIONS.map((opt) => (
          <SelectOption key={opt.value} value={opt.value}>
            {opt.label}
          </SelectOption>
        ))}
      </Select>
    </div>
  );
}
