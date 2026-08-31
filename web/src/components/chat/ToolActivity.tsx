import { cn } from "@/lib/utils";

export type ToolActivityItem = {
  id: string;
  name: string;
  state: "running" | "complete";
  context?: string;
  args?: unknown;
  result?: unknown;
  summary?: string;
  progress?: string;
  elapsedMs?: number;
};

type SafeValue = null | boolean | number | string | SafeValue[] | { [key: string]: SafeValue };

const SENSITIVE_KEY = /(pass(word)?|secret|token|api[_-]?key|authorization|cookie|credential)/i;

function redact(value: unknown, key?: string, seen = new WeakSet<object>()): SafeValue {
  if (key && SENSITIVE_KEY.test(key)) return "[redacted]";
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return `${value}n`;
  if (typeof value === "undefined") return "[unknown]";
  if (value instanceof Error) return value.message || "[unknown error]";
  if (typeof value !== "object") return `[${typeof value}]`;
  if (seen.has(value)) return "[circular]";
  seen.add(value);
  if (Array.isArray(value)) return value.map((entry) => redact(entry, undefined, seen));
  return Object.fromEntries(Object.entries(value).map(([entryKey, entryValue]) => [entryKey, redact(entryValue, entryKey, seen)]));
}

function formatDetail(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(redact(value), null, 2) ?? "[unknown]";
  } catch {
    return "[unknown]";
  }
}

function elapsedLabel(elapsedMs: number | undefined): string | null {
  if (typeof elapsedMs !== "number" || !Number.isFinite(elapsedMs) || elapsedMs < 0) return null;
  return elapsedMs < 1000 ? `${Math.round(elapsedMs)}ms` : `${(elapsedMs / 1000).toFixed(1)}s`;
}

export function ToolActivity({ item }: { item: ToolActivityItem }) {
  const elapsed = elapsedLabel(item.elapsedMs);
  const hasDetails = item.args !== undefined || item.result !== undefined;
  const stateLabel = item.state === "running" ? "Running" : "Complete";
  return (
    <article
      data-tool-id={item.id}
      data-tool-state={item.state}
      aria-label={`${item.name} tool ${item.state}`}
      className="rounded-md border border-border bg-background px-3 py-2 text-xs"
    >
      <div className="flex items-center gap-2 font-medium">
        <span role="status" aria-label={`${item.name} ${item.state}`} className={cn("inline-block h-2 w-2 rounded-full", item.state === "running" ? "animate-pulse bg-primary" : "bg-emerald-500")} />
        <span>{item.name}</span>
        <span className="text-muted-foreground">{stateLabel}</span>
        {elapsed && <span className="ml-auto font-mono text-muted-foreground">{elapsed}</span>}
      </div>
      {item.context && <div className="mt-1">{item.context}</div>}
      {item.progress && <div className="mt-1 text-muted-foreground" aria-label="Tool progress">{item.progress}</div>}
      {item.summary && <div className="mt-1 text-muted-foreground">{item.summary}</div>}
      {hasDetails && (
        <details className="mt-2">
          <summary className="cursor-pointer text-muted-foreground">Details</summary>
          <div className="mt-2 space-y-2">
            {item.args !== undefined && <div><div className="font-medium">Arguments</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words">{formatDetail(item.args)}</pre></div>}
            <div><div className="font-medium">Result</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words">{item.result === undefined ? "No result" : formatDetail(item.result)}</pre></div>
          </div>
        </details>
      )}
      {!hasDetails && item.state === "complete" && <div className="sr-only">No result</div>}
    </article>
  );
}
