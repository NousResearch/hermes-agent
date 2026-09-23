/**
 * One chat bubble — extracted from the previous in-file `MessageBubble` in
 * `SessionsPage.tsx` without behavioural changes.
 *
 * Responsibilities preserved verbatim:
 *  - role-based background / label styling (with a dedicated "Context
 *    handoff" downgrade for compaction rows, #29824)
 *  - compaction split: when the compressor merged a summary into the front
 *    of the first tail message, this component recursively renders two
 *    bubbles — labelled handoff + original reply.
 *  - search-term highlight (prefix / substring) + `data-search-hit` flag for
 *    the list's auto-scroll-to-hit behaviour.
 *  - inline rendering of attached tool calls via `ToolCallCard`.
 *
 * The component accepts the `SessionMessage` shape that the existing
 * `SessionsPage` REST path already produces. The new `TranscriptMessage`
 * projection in `components/chat/types.ts` is intentionally NOT consumed
 * here yet — wiring it in happens in Phase 2 once `ChatPage` is replaced.
 */
import { Badge } from "@nous-research/ui/ui/components/badge";

import { Markdown } from "@/components/Markdown";
import { useI18n } from "@/i18n";
import { splitCompactionContent } from "@/lib/chat/compaction";
import { timeAgo } from "@/lib/utils";
import type { SessionMessage } from "@/lib/api";

import { ToolCallCard } from "./ToolCallCard";

export interface MessageBubbleProps {
  msg: SessionMessage;
  highlight?: string;
}

export function MessageBubble({ msg, highlight }: MessageBubbleProps) {
  const { t } = useI18n();

  const ROLE_STYLES: Record<
    string,
    { bg: string; text: string; label: string }
  > = {
    user: {
      bg: "bg-primary/10",
      text: "text-primary",
      label: t.sessions.roles.user,
    },
    assistant: {
      bg: "bg-success/10",
      text: "text-success",
      label: t.sessions.roles.assistant,
    },
    system: {
      bg: "bg-muted",
      text: "text-muted-foreground",
      label: t.sessions.roles.system,
    },
    tool: {
      bg: "bg-warning/10",
      text: "text-warning",
      label: t.sessions.roles.tool,
    },
    // Compaction handoffs render as faded system-style metadata with a
    // distinctive label so they can't be mistaken for real assistant
    // replies during a scroll-back review (#29824).
    compaction: {
      bg: "bg-muted/50",
      text: "text-muted-foreground italic",
      label: "Context handoff",
    },
  };

  // When a compaction handoff is merged into the front of the first
  // tail message (the compressor's double-collision path —
  // `_merge_summary_into_tail` in `agent/context_compressor.py`),
  // the message we received is `[CONTEXT COMPACTION ...] + END_MARKER
  // + <original assistant reply>`. We split it back into two visual
  // rows here so the operator's actual answer survives as a readable
  // bubble next to the (clearly-labelled) handoff metadata (#29824).
  const compactionSplit =
    typeof msg.content === "string"
      ? splitCompactionContent(msg.content)
      : null;

  if (compactionSplit && compactionSplit.remainder) {
    return (
      <>
        <MessageBubble
          msg={{ ...msg, content: compactionSplit.summary }}
          highlight={highlight}
        />
        <MessageBubble
          msg={{
            ...msg,
            content: compactionSplit.remainder,
            // The remainder is the original assistant reply that the
            // compressor pre-pended the summary to — render with the
            // normal assistant styling, NOT the muted handoff style.
            // `isCompactionMessage` returns false on this stripped
            // content because it no longer starts with the prefix.
          }}
          highlight={highlight}
        />
      </>
    );
  }

  const isCompaction = compactionSplit !== null;
  const style = isCompaction
    ? ROLE_STYLES.compaction
    : ROLE_STYLES[msg.role] ?? ROLE_STYLES.system;
  const label = isCompaction
    ? ROLE_STYLES.compaction.label
    : msg.tool_name
      ? `${t.sessions.roles.tool}: ${msg.tool_name}`
      : style.label;

  // Check if any search term appears as a prefix of any word in content
  const isHit = (() => {
    if (!highlight || !msg.content) return false;
    const content = msg.content.toLowerCase();
    const terms = highlight.toLowerCase().split(/\s+/).filter(Boolean);
    return terms.some((term) => content.includes(term));
  })();

  // Split search query into terms for inline highlighting
  const highlightTerms =
    isHit && highlight ? highlight.split(/\s+/).filter(Boolean) : undefined;

  return (
    <div
      className={`${style.bg} p-3 ${isHit ? "ring-1 ring-warning/40" : ""}`}
      data-search-hit={isHit || undefined}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-xs font-semibold ${style.text}`}>{label}</span>
        {isHit && (
          <Badge tone="warning" className="text-xs py-0 px-1.5">
            {t.common.match}
          </Badge>
        )}
        {msg.timestamp && (
          <span className="text-xs text-text-tertiary">
            {timeAgo(msg.timestamp)}
          </span>
        )}
      </div>
      {msg.content &&
        (msg.role === "system" ? (
          <div className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">
            {msg.content}
          </div>
        ) : (
          <Markdown content={msg.content} highlightTerms={highlightTerms} />
        ))}
      {msg.tool_calls && msg.tool_calls.length > 0 && (
        <div className="mt-1">
          {msg.tool_calls.map((tc) => (
            <ToolCallCard key={tc.id} toolCall={tc} />
          ))}
        </div>
      )}
    </div>
  );
}