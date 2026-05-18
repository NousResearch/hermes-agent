import { ListItem } from "@nous-research/ui/ui/components/list-item";
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
  done: "border-border bg-muted/20",
  error: "border-destructive/50 bg-destructive/[0.04]",
};

const BULLET_TONE: Record<ToolEntry["status"], string> = {
  running: "text-primary",
  done: "text-primary/80",
  error: "text-destructive",
};

const TICK_MS = 500;

export function ToolCall({ tool }: { tool: ToolEntry }) {
  // `open` is derived: errors default-expanded, everything else collapsed.
  // `null` means "follow the default"; any explicit bool is the user's override.
  // This lets a running tool flip to expanded automatically when it errors,
  // without mirroring state in an effect.
  const [userOverride, setUserOverride] = useState<boolean | null>(null);
  const open = userOverride ?? tool.status === "error";

  // Tick `now` while the tool is running so the elapsed label updates live.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (tool.status !== "running") return;
    const id = window.setInterval(() => setNow(() => Date.now()), TICK_MS);
    return () => window.clearInterval(id);
  }, [tool.status]);

  // Historical tools (hydrated from session.resume) signal missing timestamps
  // with `startedAt === 0`; we hide the elapsed badge for those rather than
  // rendering a misleading "0ms".
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
      className={`rounded-md border overflow-hidden ${STATUS_TONE[tool.status]}`}
    >
      <ListItem
        onClick={() => setUserOverride(!open)}
        disabled={!hasBody}
        aria-expanded={open}
        className="px-2.5 py-1.5 text-xs hover:bg-foreground/2 disabled:cursor-default"
      >
        {hasBody ? (
          <Chevron className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <span className="w-3 shrink-0" />
        )}

        <Zap className={`h-3 w-3 shrink-0 ${BULLET_TONE[tool.status]}`} />

        <span className="font-mono font-medium shrink-0">{tool.name}</span>

        <span className="font-mono text-muted-foreground/80 truncate min-w-0 flex-1">
          {tool.context ?? ""}
        </span>

        {tool.status === "running" && (
          <span
            className="inline-block h-2 w-2 rounded-full bg-primary animate-pulse shrink-0"
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
            className="h-3 w-3 shrink-0 text-primary/80"
            aria-label="done"
          />
        )}

        {elapsed && (
          <span className="font-mono text-[0.65rem] text-muted-foreground tabular-nums shrink-0">
            {elapsed}
          </span>
        )}
      </ListItem>

      {open && hasBody && (
        <div className="border-t border-border/60 px-3 py-2 space-y-2 text-xs font-mono">
          {tool.context && <Section label="context">{tool.context}</Section>}

          {tool.preview && tool.status === "running" && (
            <Section label="streaming">
              {tool.preview}
              <span className="inline-block w-1.5 h-3 align-middle bg-foreground/40 ml-0.5 animate-pulse" />
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
        className={`uppercase tracking-wider text-[0.6rem] shrink-0 w-14 pt-0.5 ${
          tone === "error" ? "text-destructive/80" : "text-muted-foreground/60"
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
    return "text-emerald-500 dark:text-emerald-400";
  if (line.startsWith("-") && !line.startsWith("---"))
    return "text-destructive";
  if (line.startsWith("@@")) return "text-primary";
  return "text-muted-foreground/80";
}
