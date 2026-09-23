/**
 * Top bar of the new /chat-ui page.
 *
 * Compact, sticky header above the message list with three slots:
 *
 *   - Left: conversation title (truncated)
 *   - Right: model picker button, reasoning picker, connection indicator
 *
 * Phase 3 wires the existing {@link ModelPickerDialog}, {@link ModelInfoCard},
 * and {@link ReasoningPicker} into the header. The pickers reuse the
 * established standalone paths (REST config.set / saveConfig), matching
 * the chat sidebar's UX so the model switch persists to config.yaml and
 * applies on the next chat (the running session keeps its model until
 * rebuilt — same as the CLI surface).
 *
 * The pickers are wrapped in a local controlled state so opening one
 * doesn't fight the parent's render cycle.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Brain, ChevronDown, Cpu, Loader2, Sparkles, Wifi, WifiOff } from "lucide-react";

import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";

import { ModelInfoCard } from "@/components/ModelInfoCard";
import { ModelPickerDialog } from "@/components/ModelPickerDialog";
import type { ChatUIStore } from "@/components/chat/types";
import { api, type ModelInfoResponse } from "@/lib/api";
import {
  EFFORT_OPTIONS,
  normalizeEffort,
  VALID_EFFORTS,
} from "@/lib/reasoning-effort";

interface ChatHeaderProps {
  title: string | null;
  model: string | null;
  provider: string | null;
  reasoningEffort: string | null;
  connection: ChatUIStore["connection"];
  /**
   * Profile scope forwarded to the model + reasoning REST helpers. Both
   * pickers scope their read/write to the same profile the active chat
   * session was launched from.
   */
  profile?: string;
}

export function ChatHeader({
  title,
  model,
  reasoningEffort,
  connection,
  profile,
}: ChatHeaderProps) {
  const [modelOpen, setModelOpen] = useState(false);
  const [effort, setEffort] = useState<string>(
    normalizeEffort(reasoningEffort),
  );
  const [effortSaving, setEffortSaving] = useState(false);
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const lastInfoFetchKeyRef = useRef("");

  // Keep the local effort in sync with the live session (e.g. a
  // session.info event after a /new on the CLI side).
  useEffect(() => {
    setEffort(normalizeEffort(reasoningEffort));
  }, [reasoningEffort]);

  // Read /api/model/info so the header can:
  //   1. Display the live "effective model" badge
  //   2. Gate the ReasoningPicker on `supports_reasoning`
  // Re-fetch when the profile or the locally-tracked model changes.
  useEffect(() => {
    const key = `${profile ?? ""}:${model ?? ""}:${info?.model ?? ""}`;
    if (key === lastInfoFetchKeyRef.current) return;
    lastInfoFetchKeyRef.current = key;
    setInfoLoading(true);
    api
      .getModelInfo(profile)
      .then((r) => setInfo(r))
      .catch(() => setInfo(null))
      .finally(() => setInfoLoading(false));
  }, [profile, model, info?.model]);

  const onChangeEffort = useCallback(
    (next: string) => {
      if (!VALID_EFFORTS.has(next) || next === effort) return;
      const prev = effort;
      setEffort(next);
      setEffortSaving(true);
      api
        .saveConfig({ agent: { reasoning_effort: next } }, profile)
        .catch(() => {
          setEffort(prev);
        })
        .finally(() => setEffortSaving(false));
    },
    [effort, profile],
  );

  const supportsReasoning = !!info?.capabilities?.supports_reasoning;
  const effectiveModel = info?.model || model || "";
  const effectiveModelLabel = effectiveModel
    ? effectiveModel.split("/").slice(-1)[0]
    : "—";

  return (
    <header
      className="flex shrink-0 flex-col gap-2 border-b border-current/15 bg-background-base/95 px-3 py-2.5 backdrop-blur sm:px-5"
      data-testid="chat-ui-header"
    >
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex min-w-0 flex-1 items-baseline gap-2">
          <span className="hidden font-mono-ui text-xs uppercase tracking-wider text-text-secondary sm:inline">
            Hermes
          </span>
          {title ? (
            <h1
              className="truncate text-sm font-medium text-foreground"
              data-testid="chat-ui-session-title"
            >
              {title}
            </h1>
          ) : (
            <span
              className="truncate text-sm text-text-tertiary"
              data-testid="chat-ui-session-title-placeholder"
            >
              New conversation
            </span>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-1 text-xs text-text-secondary">
          <Button
            ghost
            size="sm"
            onClick={() => setModelOpen(true)}
            className="normal-case tracking-normal text-foreground"
            aria-label="Switch model"
            data-testid="chat-ui-model-button"
          >
            <Cpu className="mr-1 h-3.5 w-3.5" />
            <span
              className="max-w-[10rem] truncate font-mono-ui"
              data-testid="chat-ui-model"
            >
              {infoLoading && !effectiveModel ? (
                <Loader2 className="inline h-3 w-3 animate-spin" />
              ) : (
                effectiveModelLabel
              )}
            </span>
            <ChevronDown className="ml-0.5 h-3 w-3 text-text-secondary" />
          </Button>

          {supportsReasoning && (
            <div
              className="ml-1 flex items-center gap-1"
              data-testid="chat-ui-reasoning-wrap"
            >
              <Brain className="h-3.5 w-3.5 text-purple-500" />
              <Select
                aria-label="Reasoning effort"
                className="h-7 min-w-[6rem] text-xs"
                value={effort}
                onValueChange={onChangeEffort}
                disabled={effortSaving}
                data-testid="chat-ui-reasoning"
              >
                {EFFORT_OPTIONS.map((opt) => (
                  <SelectOption key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectOption>
                ))}
              </Select>
            </div>
          )}

          {!supportsReasoning && reasoningEffort && (
            <span
              className="ml-1 flex items-center gap-1 text-text-secondary"
              data-testid="chat-ui-reasoning"
            >
              <Sparkles className="h-3.5 w-3.5" />
              {reasoningEffort}
            </span>
          )}

          <ConnectionIndicator state={connection} />
        </div>
      </div>

      {info && effectiveModel && (
        <div className="border-t border-current/10 pt-2">
          <ModelInfoCard currentModel={effectiveModel} />
        </div>
      )}

      {modelOpen && (
        <ModelPickerDialog
          loader={() => api.getModelOptions(profile)}
          alwaysGlobal
          onApply={async ({ provider, model, confirmExpensiveModel }) => {
            const result = await api.setModelAssignment(
              {
                confirm_expensive_model: confirmExpensiveModel,
                scope: "main",
                provider,
                model,
              },
              profile,
            );
            return result;
          }}
          onClose={() => {
            setModelOpen(false);
            // Refresh model info so the badge + reasoning gate pick up
            // the new model.
            const key = `${profile ?? ""}:${Date.now()}`;
            lastInfoFetchKeyRef.current = "";
            void api
              .getModelInfo(profile)
              .then((r) => {
                lastInfoFetchKeyRef.current = key;
                setInfo(r);
              })
              .catch(() => undefined);
          }}
        />
      )}
    </header>
  );
}

function ConnectionIndicator({
  state,
}: {
  state: ChatUIStore["connection"];
}) {
  const isOpen = state === "open";
  return (
    <span
      className="ml-1 flex items-center gap-1 text-text-tertiary"
      aria-label={`Gateway ${state}`}
      data-testid="chat-ui-connection"
    >
      {isOpen ? (
        <Wifi className="h-3.5 w-3.5 text-success" />
      ) : (
        <WifiOff className="h-3.5 w-3.5 text-text-tertiary" />
      )}
      <span className="hidden sm:inline">{state}</span>
    </span>
  );
}
