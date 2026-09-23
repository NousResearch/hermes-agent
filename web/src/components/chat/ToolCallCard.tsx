/**
 * Single tool-call card used inside a `MessageBubble`.
 *
 * Behaviour + styles preserved verbatim from the previous in-file
 * `ToolCallBlock` in `SessionsPage.tsx`; renamed to `ToolCallCard` for the
 * extracted `components/chat/` namespace. Renders the tool name with a
 * collapse/expand chevron and pretty-prints the arguments JSON.
 *
 * The card accepts the projection shape from `components/chat/types.ts` —
 * either a raw REST `{ id, function: { name, arguments } }` row (existing
 * `SessionMessage.tool_calls` item) or the gateway `ChatToolCall` projection.
 * Both shapes carry `id`, a display name, and the arguments (raw or
 * pre-decoded). When `argumentsDecoded` is supplied, that wins; otherwise the
 * raw string is pretty-printed.
 */
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { useI18n } from "@/i18n";

import type { ChatToolCall } from "./types";

export interface ToolCallCardProps {
  /** REST shape (existing): `{ id, function: { name, arguments } }`. */
  toolCall?:
    | { id: string; function: { name: string; arguments: string } }
    | ChatToolCall;
  /** New projection shape (gateway). Used when `toolCall` is omitted. */
  call?: ChatToolCall;
}

export function ToolCallCard({ toolCall, call }: ToolCallCardProps) {
  const [open, setOpen] = useState(false);
  const { t } = useI18n();

  // Normalise both shapes into `{ id, name, arguments }`.
  const normalised = (() => {
    if (call) {
      const args = call.argumentsDecoded
        ? JSON.stringify(call.argumentsDecoded, null, 2)
        : call.argumentsRaw ?? "";
      return { id: call.id, name: call.name, args };
    }
    if (toolCall && "function" in toolCall) {
      let args = toolCall.function.arguments;
      try {
        args = JSON.stringify(JSON.parse(args), null, 2);
      } catch {
        // keep as-is
      }
      return {
        id: toolCall.id,
        name: toolCall.function.name,
        args,
      };
    }
    return null;
  })();

  if (!normalised) return null;
  const { id, name, args } = normalised;

  return (
    <div className="mt-2 border border-warning/20 bg-warning/5">
      <ListItem
        onClick={() => setOpen(!open)}
        aria-label={`${open ? t.common.collapse : t.common.expand} tool call ${name}`}
        aria-expanded={open}
        className="px-3 py-2 text-xs text-warning hover:bg-warning/10 hover:text-warning"
      >
        {open ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <span className="font-mono-ui font-medium">{name}</span>
        <span className="text-warning/50 ml-auto">{id}</span>
      </ListItem>
      {open && (
        <pre className="border-t border-warning/20 px-3 py-2 text-xs text-warning/80 overflow-x-auto whitespace-pre-wrap font-mono">
          {args}
        </pre>
      )}
    </div>
  );
}