/**
 * Assistant turn rendered by the new structured Chat UI.
 *
 * Reads a {@link ChatUIAssistantTurn} and lays out:
 *   - streaming text (with a flashing caret while open)
 *   - Copy action (always available)
 *   - Regenerate action (disabled — see below)
 *   - reasoning section when reasoning text is present
 *   - tool cards in order
 *   - usage line when message.complete supplied it
 *   - status / failure-reason on error
 *
 * Regenerate: NOT YET SUPPORTED.
 *
 * The verified gateway contract (apps/shared/src/gateway-contract.generated.ts
 * — see the `RpcMethods` map near line ~4451) does NOT expose a regenerate
 * or resend method. The closest analog is `session.redirect`, which is a
 * mid-flight steer for the *current* turn, not a "redo the last turn".
 * `prompt.submit` accepts `truncate_before_user_ordinal` /
 * `truncate_before_row_id` / `truncate_before_message_id` (with
 * `confirm_truncate` / `confirm_empty_truncate`) for a destructive rewind,
 * but no first-class regenerate that preserves the user message and only
 * re-runs the assistant turn.
 *
 * Phase 3: render a disabled regenerate button with a tooltip explaining
 * the contract gap. Phase 4 / a follow-up can add it once the gateway
 * surfaces a `session.regenerate` (or equivalent) RPC.
 *
 * Phase 4 polish: stricter typography (role label uppercase mono, body
 * text relaxed leading), clearer streaming badge, accessible tooltips,
 * always-visible copy button (not hover-gated) for keyboard users.
 */

import { Check, Copy, Loader2, RefreshCw } from "lucide-react";
import { useState } from "react";

import { Markdown } from "@/components/Markdown";
import { ChatToolCard } from "@/components/chat/ChatToolCard";
import { ReasoningPanel } from "@/components/chat/ReasoningPanel";
import type { ChatUIAssistantTurn } from "@/components/chat/types";
import { copyTextToClipboard } from "@/lib/clipboard";
import { timeAgo } from "@/lib/utils";

interface AssistantMessageProps {
  turn: ChatUIAssistantTurn;
}

const STATUS_LABEL: Record<NonNullable<ChatUIAssistantTurn["status"]>, string> = {
  complete: "Complete",
  error: "Error",
  interrupted: "Interrupted",
};

// Documented in the file header — copied here for the tooltip + disabled
// reason. Keep in sync with the contract scan; this is the source of truth
// for the Chat UI's regenerate behaviour until the gateway ships it.
const REGENERATE_UNSUPPORTED_REASON =
  "Regenerate is not supported by the verified gateway contract (no session.regenerate RPC).";

export function AssistantMessage({ turn }: AssistantMessageProps) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    const ok = await copyTextToClipboard(turn.text);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  const toolList = Object.values(turn.tools);
  const hasReasoning = turn.reasoning.trim().length > 0;
  const hasFailure =
    turn.failureReason !== undefined && turn.failureReason !== null && turn.failureReason.length > 0;
  const usageLine = formatUsage(
    typeof turn.usage?.total_tokens === "number"
      ? turn.usage.total_tokens
      : null,
  );
  // Regenerate is NOT YET supported by the verified gateway contract —
  // see file header. The button is rendered disabled with an explanatory
  // tooltip so the affordance stays discoverable but the call is a no-op.
  const regenerateDisabled = true;

  return (
    <article
      className="group/turn flex flex-col gap-3 rounded-lg border border-current/10 bg-card px-4 py-3"
      data-role="assistant"
      data-streaming={turn.streaming || undefined}
    >
      <header className="flex items-center gap-2 text-xs">
        <span className="flex items-center gap-1.5 font-mono-ui text-xs uppercase tracking-wider text-success">
          Assistant
          {turn.streaming && (
            <span
              className="flex items-center gap-1 rounded bg-midground/20 px-1.5 py-0.5 text-text-secondary"
              data-testid="chat-ui-streaming-badge"
              aria-label="Streaming"
            >
              <Loader2 className="h-3 w-3 animate-spin" />
              Streaming
            </span>
          )}
        </span>
        {turn.status && !turn.streaming && (
          <span className="text-text-tertiary">{STATUS_LABEL[turn.status]}</span>
        )}
        {turn.finalizedAt && !turn.streaming && (
          <span className="text-text-tertiary" title={new Date(turn.finalizedAt * 1000).toLocaleString()}>
            {timeAgo(turn.finalizedAt)}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            disabled={regenerateDisabled}
            onClick={() => {
              // Intentionally a no-op: regenerate is not yet exposed by
              // the gateway contract. The button is kept visible so the
              // affordance is discoverable and the disabled state surfaces
              // the contract gap.
            }}
            aria-label="Regenerate response (not yet supported by the gateway)"
            aria-disabled="true"
            title={REGENERATE_UNSUPPORTED_REASON}
            data-testid="chat-ui-regenerate"
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-text-tertiary opacity-80 transition-colors hover:bg-midground/10 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw className="h-3 w-3" aria-hidden />
            <span className="hidden sm:inline">Regenerate</span>
          </button>
          <button
            type="button"
            onClick={onCopy}
            aria-label="Copy assistant message"
            aria-pressed={copied}
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-text-tertiary opacity-80 transition-colors hover:bg-midground/10 hover:text-foreground focus:bg-midground/10 focus:text-foreground focus:opacity-100 focus:outline-none"
            data-testid="chat-ui-copy"
          >
            {copied ? (
              <>
                <Check className="h-3 w-3 text-success" aria-hidden />
                <span className="hidden sm:inline">Copied</span>
              </>
            ) : (
              <>
                <Copy className="h-3 w-3" aria-hidden />
                <span className="hidden sm:inline">Copy</span>
              </>
            )}
          </button>
        </div>
      </header>

      {turn.text.length > 0 && (
        <div
          className="text-[14px] leading-6 text-foreground"
          data-testid="chat-ui-assistant-text"
        >
          <Markdown content={turn.text} streaming={turn.streaming} />
        </div>
      )}

      {hasReasoning && (
        <ReasoningPanel text={turn.reasoning} initiallyOpen={turn.streaming} />
      )}

      {toolList.length > 0 && (
        <div className="flex flex-col gap-2">
          {toolList.map((tool) => (
            <ChatToolCard key={tool.toolId} tool={tool} />
          ))}
        </div>
      )}

      {hasFailure && (
        <div
          className="rounded border border-destructive/30 bg-destructive/10 px-2 py-1 text-xs text-destructive"
          role="alert"
        >
          {turn.failureReason}
        </div>
      )}

      {usageLine && (
        <footer className="text-xs text-text-tertiary">{usageLine}</footer>
      )}
    </article>
  );
}

function formatUsage(totalTokens?: number | null): string | null {
  if (typeof totalTokens !== "number") return null;
  return `${totalTokens.toLocaleString()} tokens`;
}
