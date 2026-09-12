import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { ListItem } from "@nous-research/ui/ui/components/list-item";

import { Markdown } from "@/components/Markdown";
import { useI18n } from "@/i18n";
import type { SessionMessage } from "@/lib/api-types";
import { timeAgo } from "@/lib/utils";

function ToolCallBlock({
  toolCall,
}: {
  toolCall: { id: string; function: { name: string; arguments: string } };
}) {
  const [open, setOpen] = useState(false);
  const { t } = useI18n();

  let args = toolCall.function.arguments;
  try {
    args = JSON.stringify(JSON.parse(args), null, 2);
  } catch {
    // keep as-is
  }

  return (
    <div className="mt-2 border border-warning/20 bg-warning/5">
      <ListItem
        onClick={() => setOpen(!open)}
        aria-label={`${open ? t.common.collapse : t.common.expand} tool call ${toolCall.function.name}`}
        aria-expanded={open}
        className="px-3 py-2 text-xs text-warning hover:bg-warning/10 hover:text-warning"
      >
        {open ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <span className="font-mono-ui font-medium">
          {toolCall.function.name}
        </span>
        <span className="text-warning/50 ml-auto">{toolCall.id}</span>
      </ListItem>
      {open && (
        <pre className="border-t border-warning/20 px-3 py-2 text-xs text-warning/80 overflow-x-auto whitespace-pre-wrap font-mono">
          {args}
        </pre>
      )}
    </div>
  );
}

// Context-compaction handoff blocks are persisted as ``role="user"`` or
// ``role="assistant"`` with content starting with one of these prefixes —
// they're metadata inserted by ``agent/context_compressor.py``, NOT real
// turns the user typed or the model replied with. Rendering them with
// the same styling as regular messages confuses operators scrolling the
// session timeline (#29824 — "WebUI can show context compaction block
// instead of latest assistant response after compression"), so we
// detect them here and downgrade them to a muted, clearly-labelled
// "Context handoff" row.
//
// Keep these prefixes (and the END marker below) in sync with
// ``SUMMARY_PREFIX`` / ``LEGACY_SUMMARY_PREFIX`` and the
// merge-into-tail marker in ``agent/context_compressor.py``.
const COMPACTION_PREFIXES = [
  "[CONTEXT COMPACTION — REFERENCE ONLY]",
  "[CONTEXT COMPACTION - REFERENCE ONLY]",
  "[CONTEXT SUMMARY]:",
] as const;

// Marker the compressor inserts between a merged summary and the
// original tail message content. When the summary role would collide
// with both head and tail roles (e.g. head ends with ``user`` and tail
// starts with ``assistant``), the compressor merges the summary as a
// prefix on the first tail message instead of inserting a standalone
// row. We split on this marker so the WebUI still shows the original
// assistant reply as its own readable bubble — otherwise the merged
// row reads as a single opaque "Context compaction" block and the
// user can't see the reply (#29824).
const COMPACTION_END_MARKER =
  "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---";

interface CompactionSplit {
  /** Summary text (header + body, without the end marker). */
  summary: string;
  /** Original message content that came after the end marker. */
  remainder: string;
}

function splitCompactionContent(content: string): CompactionSplit | null {
  const head = content.trimStart();
  if (!COMPACTION_PREFIXES.some((p) => head.startsWith(p))) return null;
  const markerIdx = content.indexOf(COMPACTION_END_MARKER);
  if (markerIdx < 0) {
    return { summary: content, remainder: "" };
  }
  return {
    summary: content.slice(0, markerIdx),
    remainder: content
      .slice(markerIdx + COMPACTION_END_MARKER.length)
      .replace(/^\s+/, ""),
  };
}


function MessageBubble({
  msg,
  highlight,
}: {
  msg: SessionMessage;
  highlight?: string;
}) {
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
  // ``_merge_summary_into_tail`` in ``agent/context_compressor.py``),
  // the message we received is ``[CONTEXT COMPACTION ...] + END_MARKER
  // + <original assistant reply>``. We split it back into two visual
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
            // ``isCompactionMessage`` returns false on this stripped
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
            <ToolCallBlock key={tc.id} toolCall={tc} />
          ))}
        </div>
      )}
    </div>
  );
}

/** Message list with auto-scroll to first search hit. */
export function MessageList({
  messages,
  highlight,
}: {
  messages: SessionMessage[];
  highlight?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!highlight || !containerRef.current) return;
    // Scroll to first hit after render
    const timer = setTimeout(() => {
      const hit = containerRef.current?.querySelector("[data-search-hit]");
      if (hit) {
        hit.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 50);
    return () => clearTimeout(timer);
  }, [messages, highlight]);

  return (
    <div
      ref={containerRef}
      className="flex flex-col gap-3 max-h-[600px] overflow-y-auto pr-2"
    >
      {messages.map((msg, i) => (
        <MessageBubble key={i} msg={msg} highlight={highlight} />
      ))}
    </div>
  );
}
