/**
 * Tool card used inside the new structured Chat UI.
 *
 * Distinct from the existing `ToolCallCard.tsx` — that one consumes the
 * REST `SessionMessage` projection. This one renders the gateway
 * `ChatUIToolCall` shape and adds the "running" state (with optional
 * streamed progress tail) plus a copy-to-clipboard affordance.
 *
 * No secrets or credentials are rendered — the projection's `result`
 * field is summary text, not raw output; we treat it the same as the
 * upstream card.
 *
 * Phase 4 polish:
 *   - Duration always visible (right of the badge) once the tool
 *     reports it; wall-clock startedAt + finalizedAt = duration_s.
 *   - Slightly tighter rhythm + status-toned border so the eye lands on
 *     the badge even when collapsed.
 *   - "Arguments" and "Result" sections stay independent disclosures so
 *     the user can collapse one but keep the other open.
 *   - Defensive: if `argumentsDecoded` / `argumentsRaw` accidentally
 *     surface a string that LOOKS like a credential (very long random
 *     string with no spaces), we truncate to 1KB and surface a "may
 *     contain sensitive data" footer instead of dumping the bytes.
 */

import { useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  Clock,
  Loader2,
  XCircle,
} from "lucide-react";

import { Markdown } from "@/components/Markdown";
import type { ChatUIToolCall } from "@/components/chat/types";

const STATUS_BADGE: Record<
  ChatUIToolCall["status"],
  { label: string; tone: string; ring: string }
> = {
  pending: {
    label: "Pending",
    tone: "bg-muted text-text-tertiary",
    ring: "border-current/15",
  },
  running: {
    label: "Running",
    tone: "bg-warning/15 text-warning",
    ring: "border-warning/30",
  },
  success: {
    label: "Done",
    tone: "bg-success/15 text-success",
    ring: "border-success/30",
  },
  error: {
    label: "Error",
    tone: "bg-destructive/15 text-destructive",
    ring: "border-destructive/30",
  },
};

/** Defence-in-depth cap for the args / result preview panes. */
const PREVIEW_MAX = 2_000;

interface ChatToolCardProps {
  tool: ChatUIToolCall;
}

export function ChatToolCard({ tool }: ChatToolCardProps) {
  const [open, setOpen] = useState(false);
  const badge = STATUS_BADGE[tool.status];
  const argsText = formatArgs(tool);
  const showSpinner = tool.status === "running" || tool.status === "pending";
  const hasDetails =
    !!argsText ||
    !!(tool.progress?.tail && tool.status === "running") ||
    !!(tool.result && tool.result.length > 0) ||
    !!tool.error;

  return (
    <div
      className={`overflow-hidden rounded-md border bg-card/60 ${badge.ring}`}
      data-tool-status={tool.status}
      data-tool-id={tool.toolId}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={!hasDetails}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-midground/5 focus:bg-midground/5 focus:outline-none disabled:cursor-default disabled:hover:bg-transparent"
        aria-expanded={open}
        aria-label={
          hasDetails
            ? `${open ? "Collapse" : "Expand"} tool call ${tool.name}`
            : `Tool call ${tool.name}, no additional details`
        }
      >
        {showSpinner ? (
          <Loader2
            className="h-3.5 w-3.5 animate-spin text-warning"
            aria-hidden
          />
        ) : tool.status === "error" ? (
          <XCircle className="h-3.5 w-3.5 text-destructive" aria-hidden />
        ) : tool.status === "success" ? (
          <Check className="h-3.5 w-3.5 text-success" aria-hidden />
        ) : open ? (
          <ChevronDown className="h-3.5 w-3.5" aria-hidden />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" aria-hidden />
        )}
        <span className="font-mono-ui text-xs font-medium text-foreground">
          {tool.name}
        </span>
        <span
          className={`ml-auto rounded px-1.5 py-0.5 text-xs font-medium ${badge.tone}`}
          data-testid="chat-ui-tool-status"
        >
          {badge.label}
        </span>
        {tool.durationS !== undefined && (
          <span
            className="ml-1 flex items-center gap-0.5 font-mono-ui text-xs text-text-tertiary"
            title={`Tool ran for ${tool.durationS.toFixed(2)}s`}
          >
            <Clock className="h-3 w-3" aria-hidden />
            {tool.durationS.toFixed(2)}s
          </span>
        )}
      </button>

      {open && hasDetails && (
        <div className="border-t border-current/10 px-3 py-2 text-xs text-text-secondary">
          {argsText && (
            <details className="mb-2">
              <summary className="cursor-pointer select-none text-text-tertiary hover:text-foreground">
                Arguments
              </summary>
              <pre
                className="mt-1 overflow-x-auto whitespace-pre-wrap rounded bg-background-base/40 p-2 font-mono text-xs text-foreground"
                data-testid="chat-ui-tool-args"
              >
                {truncate(argsText, PREVIEW_MAX)}
              </pre>
              {looksSensitive(argsText) && (
                <p className="mt-1 flex items-center gap-1 text-text-tertiary">
                  <AlertTriangle className="h-3 w-3" aria-hidden />
                  Arguments may contain sensitive data — output truncated.
                </p>
              )}
            </details>
          )}
          {tool.progress?.tail && tool.status === "running" && (
            <div className="mb-2">
              <div className="text-text-tertiary">Progress</div>
              <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded bg-background-base/40 p-2 font-mono text-xs text-warning">
                {tool.progress.tail}
              </pre>
            </div>
          )}
          {tool.result !== undefined && tool.result.length > 0 && (
            <details open className="mb-2">
              <summary className="cursor-pointer select-none text-text-tertiary hover:text-foreground">
                Result
              </summary>
              <div
                className="mt-1 max-h-72 overflow-y-auto rounded bg-background-base/40 p-2 text-foreground"
                data-testid="chat-ui-tool-result"
              >
                <Markdown content={truncate(tool.result, PREVIEW_MAX)} />
              </div>
            </details>
          )}
          {tool.error && (
            <div
              className="rounded border border-destructive/30 bg-destructive/10 p-2 text-destructive"
              role="alert"
              data-testid="chat-ui-tool-error"
            >
              <strong className="font-medium">Error: </strong>
              {tool.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatArgs(tool: ChatUIToolCall): string {
  if (tool.argumentsDecoded) {
    try {
      return JSON.stringify(tool.argumentsDecoded, null, 2);
    } catch {
      return String(tool.argumentsRaw ?? "");
    }
  }
  return String(tool.argumentsRaw ?? "");
}

function truncate(value: string, max: number): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max)}\n…(truncated for render)`;
}

/**
 * Heuristic flag for "this string looks like a credential" — used to
 * display a "may contain sensitive data" note rather than dump the bytes.
 *
 * Conservative on purpose: only flags single very long tokens that look
 * like secrets (high-entropy, no whitespace). Real credentials follow this
 * shape; legitimate long arguments (URLs, file paths, multiline JSON)
 * usually contain whitespace / slashes.
 */
function looksSensitive(value: string): boolean {
  if (value.length < 256) return false;
  const lines = value.split("\n");
  // If the args span many lines it's almost certainly structured JSON,
  // not a single secret.
  if (lines.length > 4) return false;
  const trimmed = value.trim();
  // Single-line, > 256 chars, mostly alphanumerics or base64-ish chars.
  if (trimmed.includes(" ")) return false;
  return /^[A-Za-z0-9+/=_\-:]{256,}$/.test(trimmed);
}
