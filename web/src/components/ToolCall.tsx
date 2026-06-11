import {
  AlertCircle,
  Check,
  ChevronDown,
  ChevronRight,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";

/**
 * Expandable tool call row — the web equivalent of Ink's ToolTrail node.
 *
 * Renders one `tool.start` + `tool.complete` pair (plus any `tool.progress`
 * in between) as a single collapsible item in the transcript:
 *
 *   ▸ ● read_file(path=/foo)                         2.3s
 *
 * Click the header to reveal a preformatted body with context (args), the
 * streaming preview (while running), and the final summary or error. Error
 * rows auto-expand so failures aren't silently collapsed.
 */

export interface ToolEntry {
  kind: "tool";
  id: string;
  tool_id: string;
  name: string;
  context?: string;
  preview?: string;
  summary?: string;
  error?: string;
  inline_diff?: string;
  status: "running" | "done" | "error";
  startedAt: number;
  completedAt?: number;
}

const STATUS_TONE: Record<ToolEntry["status"], string> = {
  running: "border-primary/40 bg-primary/[0.04]",
  done: "border-border/30 bg-muted/10",
  error: "border-destructive/40 bg-destructive/[0.04]",
};

const BULLET_TONE: Record<ToolEntry["status"], string> = {
  running: "text-primary",
  done: "text-primary/70",
  error: "text-destructive",
};

const TICK_MS = 500;

export function ToolCall({ tool }: { tool: ToolEntry }) {
  // `open` is derived: errors default-expanded, everything else collapsed.
  const [userOverride, setUserOverride] = useState<boolean | null>(null);
  const open = userOverride ?? tool.status === "error";

  // Tick `now` while the tool is running so the elapsed label updates live.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (tool.status !== "running") return;
    const id = window.setInterval(() => setNow(() => Date.now()), TICK_MS);
    return () => window.clearInterval(id);
  }, [tool.status]);

  const hasTimestamps = tool.startedAt > 0;
  const elapsed = hasTimestamps
    ? fmtElapsed((tool.completedAt ?? now) - tool.startedAt)
    : null;

  const hasBody = !!(
    tool.context ||
    tool.preview ||
    tool.summary ||
    tool.error ||
    tool.inline_diff
  );

  const Chevron = open ? ChevronDown : ChevronRight;

  return (
    <div
      className={`rounded-lg border overflow-hidden transition-colors ${STATUS_TONE[tool.status]}`}
    >
      <button
        onClick={() => setUserOverride(!open)}
        disabled={!hasBody}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-xs hover:bg-foreground/5 disabled:cursor-default transition-colors"
      >
        {hasBody ? (
          <Chevron className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <span className="w-3 shrink-0" />
        )}

        <Zap className={`h-3 w-3 shrink-0 ${BULLET_TONE[tool.status]}`} />

        <span className="font-mono font-medium shrink-0">{tool.name}</span>

        {tool.context && (
          <span className="font-mono text-text-secondary truncate min-w-0 flex-1">
            {tool.context}
          </span>
        )}

        {tool.status === "running" && (
          <span
            className="inline-block h-1.5 w-1.5 rounded-full bg-primary animate-pulse shrink-0"
            title="running"
          />
        )}
        {tool.status === "error" && (
          <AlertCircle
            className="h-3 w-3 shrink-0 text-destructive"
            aria-label="error"
          />
        )}
        {tool.status === "done" && (
          <Check
            className="h-3 w-3 shrink-0 text-primary/70"
            aria-label="done"
          />
        )}

        {elapsed && (
          <span className="font-mono text-xs text-text-tertiary tabular-nums shrink-0">
            {elapsed}
          </span>
        )}
      </button>

      {open && hasBody && (
        <div className="border-t border-border/30 px-3 py-2.5 space-y-2 text-xs font-mono">
          {tool.context && <Section label="context">{tool.context}</Section>}

          {tool.preview && tool.status === "running" && (
            <Section label="streaming">
              {tool.preview}
              <span className="inline-block w-1 h-2.5 align-middle bg-foreground/40 ml-0.5 animate-pulse" />
            </Section>
          )}

          {tool.inline_diff && (
            <Section label="diff">
              <pre className="whitespace-pre overflow-x-auto text-[0.7rem] leading-snug">
                {colorizeDiff(tool.inline_diff)}
              </pre>
            </Section>
          )}

          {tool.summary && (
            <Section label="result">
              <span className="text-foreground/90 whitespace-pre-wrap">
                {tool.summary}
              </span>
            </Section>
          )}

          {tool.error && (
            <Section label="error" tone="error">
              <span className="text-destructive whitespace-pre-wrap">
                {tool.error}
              </span>
            </Section>
          )}
        </div>
      )}
    </div>
  );
}

function Section({
  label,
  children,
  tone,
}: {
  label: string;
  children: React.ReactNode;
  tone?: "error";
}) {
  return (
    <div className="flex gap-3">
      <span
        className={`text-xs font-medium tracking-wide shrink-0 w-16 pt-0.5 ${
          tone === "error" ? "text-destructive" : "text-text-tertiary"
        }`}
      >
        {label}
      </span>

      <div className="flex-1 min-w-0 text-muted-foreground">{children}</div>
    </div>
  );
}

function fmtElapsed(ms: number): string {
  const sec = Math.max(0, ms) / 1000;
  if (sec < 1) return `${Math.round(ms)}ms`;
  if (sec < 10) return `${sec.toFixed(1)}s`;
  if (sec < 60) return `${Math.round(sec)}s`;

  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return s ? `${m}m ${s}s` : `${m}m`;
}

/** Colorize unified-diff lines for the inline diff section. */
function colorizeDiff(diff: string): React.ReactNode {
  return diff.split("\n").map((line, i) => (
    <div key={i} className={diffLineClass(line)}>
      {line || "\u00A0"}
    </div>
  ));
}

function diffLineClass(line: string): string {
  if (line.startsWith("+") && !line.startsWith("+++"))
    return "text-success";
  if (line.startsWith("-") && !line.startsWith("---"))
    return "text-destructive";
  if (line.startsWith("@@")) return "text-primary";
  return "text-text-secondary";
}
