import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  FileText,
  Loader2,
  Sparkles,
  Wrench,
} from "lucide-react";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  type SubagentNode,
  type SubagentProgress,
  type SubagentStatus,
  type SubagentStreamEntry,
  activeSubagentCount,
  buildSubagentTree,
  getSubagentsBySession,
  subscribe,
} from "@/store/subagents";

// ── Status glyph ──────────────────────────────────────────────────────

function StatusGlyph({
  status,
  label,
}: {
  status: SubagentStatus;
  label: { running: string; done: string; failed: string };
}) {
  if (status === "running" || status === "queued") {
    return (
      <Loader2
        aria-label={label.running}
        className="size-3.5 shrink-0 animate-spin text-muted-foreground/80"
      />
    );
  }
  if (status === "failed" || status === "interrupted") {
    return (
      <AlertCircle
        aria-label={label.failed}
        className="size-3.5 shrink-0 text-destructive"
      />
    );
  }
  return (
    <CheckCircle2
      aria-label={label.done}
      className="size-3.5 shrink-0 text-emerald-600/85 dark:text-emerald-400/85"
    />
  );
}

// ── Formatters ────────────────────────────────────────────────────────

function fmtDuration(seconds: number | undefined): string {
  if (!seconds || seconds <= 0) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function fmtTokens(value: number | undefined): string {
  if (!value) return "";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M tok`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k tok`;
  return `${value} tok`;
}

function fmtAge(updatedAt: number, nowMs: number): string {
  const s = Math.max(0, Math.round((nowMs - updatedAt) / 1000));
  if (s < 2) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

// ── Stream tone ───────────────────────────────────────────────────────

const STREAM_TONE: Record<SubagentStreamEntry["kind"], string> = {
  progress: "text-muted-foreground/75",
  summary: "text-foreground/85",
  thinking: "text-muted-foreground/80",
  tool: "text-foreground/85",
};

function streamGlyph(entry: SubagentStreamEntry) {
  if (entry.isError) {
    return <AlertCircle aria-hidden className="mt-0.5 size-3 shrink-0 text-destructive" />;
  }
  if (entry.kind === "tool") {
    return <span aria-hidden className="mt-0.5 size-1.5 shrink-0 rounded-full bg-foreground/55" />;
  }
  if (entry.kind === "summary") {
    return <CheckCircle2 aria-hidden className="mt-0.5 size-3 shrink-0 text-emerald-600/85 dark:text-emerald-400/85" />;
  }
  if (entry.kind === "thinking") {
    return <span aria-hidden className="font-mono text-[0.7rem] leading-none text-muted-foreground/70">…</span>;
  }
  return <span aria-hidden className="mt-0.5 size-1 shrink-0 rounded-full bg-muted-foreground/55" />;
}

// ── Tree helpers ──────────────────────────────────────────────────────

const flatten = (nodes: readonly SubagentNode[]): SubagentNode[] =>
  nodes.flatMap((node) => [node, ...flatten(node.children)]);

interface RootGroup {
  id: string;
  delegationIndex: number;
  nodes: SubagentNode[];
  taskCount: number;
}

function groupDelegations(roots: readonly SubagentNode[]): RootGroup[] {
  const groups: RootGroup[] = [];
  let n = 0;
  for (const node of roots) {
    const prev = groups.at(-1);
    const prevTail = prev?.nodes.at(-1);
    const closeInTime = prevTail
      ? Math.abs(node.startedAt - prevTail.startedAt) <= 5_000
      : false;
    const sameShape =
      prev && node.taskCount > 1 && prev.taskCount === node.taskCount;
    const uniqueStep = prev
      ? !prev.nodes.some((item) => item.taskIndex === node.taskIndex)
      : false;

    if (prev && sameShape && closeInTime && uniqueStep) {
      prev.nodes.push(node);
      continue;
    }
    if (node.taskCount > 1) {
      n += 1;
      groups.push({
        id: `delegation-${n}`,
        delegationIndex: n,
        nodes: [node],
        taskCount: node.taskCount,
      });
      continue;
    }
    groups.push({
      id: node.id,
      delegationIndex: 0,
      nodes: [node],
      taskCount: node.taskCount,
    });
  }
  return groups;
}

// ── Sub-components ────────────────────────────────────────────────────

function StreamLine({
  active,
  entry,
}: {
  active: boolean;
  entry: SubagentStreamEntry;
}) {
  const isMono = entry.kind === "tool";
  const tone = entry.isError ? "text-destructive" : STREAM_TONE[entry.kind];

  return (
    <div className="flex min-w-0 items-baseline gap-2 text-[0.72rem] leading-relaxed">
      <span className="flex h-[0.95rem] shrink-0 items-center">
        {streamGlyph(entry)}
      </span>
      <span
        className={cn(
          "min-w-0 flex-1 wrap-anywhere",
          tone,
          isMono && "font-mono text-[0.69rem]",
        )}
      >
        {entry.text}
        {active ? (
          <Loader2 className="ml-1 inline-block size-2.5 animate-spin align-middle text-muted-foreground/70" />
        ) : null}
      </span>
    </div>
  );
}

function SubagentRow({
  node,
  depth = 0,
  nowMs,
}: {
  node: SubagentNode;
  depth?: number;
  nowMs: number;
}) {
  const agents = useI18n().t.agents;
  const running = node.status === "running" || node.status === "queued";
  const [open, setOpen] = useState(() => running || depth < 2);

  useEffect(() => {
    if (running) setOpen(true);
  }, [running]);

  const durationSeconds =
    typeof node.durationSeconds === "number"
      ? Math.max(0, Math.round(node.durationSeconds))
      : 0;

  const visibleRows = open ? node.stream.slice(-10) : node.stream.slice(-2);
  const fileLines = [
    ...node.filesWritten.map((p) => `+ ${p}`),
    ...node.filesRead.map((p) => `· ${p}`),
  ];

  const subtitle = [
    node.model,
    fmtDuration(durationSeconds),
    node.toolCount ? agents.toolsCount(node.toolCount) : "",
    fmtTokens((node.inputTokens ?? 0) + (node.outputTokens ?? 0)),
    agents.updatedAgo(fmtAge(node.updatedAt, nowMs, agents)),
  ].filter(Boolean);

  return (
    <div
      className={cn("grid min-w-0 max-w-full gap-2", depth > 0 && "pl-4")}
    >
      <button
        aria-expanded={open}
        className="group flex w-full min-w-0 items-start gap-2.5 text-left"
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <span className="mt-0.5 flex h-[1.1rem] shrink-0 items-center">
          <StatusGlyph status={node.status} label={agents} />
        </span>
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span
            className={cn(
              "wrap-anywhere text-[0.82rem] font-medium leading-[1.1rem] text-foreground/90 transition-colors group-hover:text-foreground",
              running && "text-foreground/65",
            )}
          >
            {node.goal}
          </span>
          {subtitle.length > 0 ? (
            <span className="text-[0.66rem] leading-[1.05rem] text-muted-foreground/65">
              {subtitle.join(" · ")}
            </span>
          ) : null}
        </span>
        <span className="mt-1 shrink-0 text-[0.6rem] text-muted-foreground/60">
          {open ? (
            <ChevronDown className="size-3" />
          ) : (
            <ChevronRight className="size-3" />
          )}
        </span>
      </button>

      {open && visibleRows.length > 0 ? (
        <div className="grid min-w-0 gap-1 pl-6">
          {visibleRows.map((entry, i) => (
            <StreamLine
              active={running && i === visibleRows.length - 1}
              entry={entry}
              key={`${entry.kind}:${entry.at}:${i}`}
            />
          ))}
        </div>
      ) : null}

      {open && fileLines.length > 0 ? (
        <div className="grid min-w-0 gap-0.5 pl-6">
          <p className="text-[0.58rem] font-medium tracking-wider text-muted-foreground/60 uppercase">
            {agents.files}
          </p>
          {fileLines.slice(0, 8).map((line) => (
            <p
              className="wrap-break-word font-mono text-[0.67rem] leading-relaxed text-muted-foreground/80"
              key={line}
            >
              {line}
            </p>
          ))}
          {fileLines.length > 8 ? (
            <p className="font-mono text-[0.67rem] leading-relaxed text-muted-foreground/65">
              {agents.moreFiles(fileLines.length - 8)}
            </p>
          ) : null}
        </div>
      ) : null}

      {node.children.length > 0 ? (
        <div className="grid min-w-0 gap-3 pl-6">
          {node.children.map((child) => (
            <SubagentRow
              depth={depth + 1}
              key={child.id}
              node={child}
              nowMs={nowMs}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function DelegationGroup({
  group,
  nowMs,
}: {
  group: RootGroup;
  nowMs: number;
}) {
  const agents = useI18n().t.agents;

  if (group.nodes.length === 1 && group.taskCount <= 1) {
    return <SubagentRow node={group.nodes[0]!} nowMs={nowMs} />;
  }

  const activeWorkers = group.nodes.filter(
    (n) => n.status === "running" || n.status === "queued",
  ).length;

  return (
    <section className="grid min-w-0 gap-3">
      <p className="text-[0.66rem] font-medium uppercase tracking-wider text-muted-foreground/70">
        {group.delegationIndex > 0
          ? agents.delegation(group.delegationIndex)
          : ""}{" "}
        <span className="text-muted-foreground/50">·</span>{" "}
        {agents.workers(group.nodes.length)}
        {activeWorkers > 0 ? (
          <span className="text-primary/85">
            {" "}
            · {agents.workersActive(activeWorkers)}
          </span>
        ) : null}
      </p>
      <div className="grid min-w-0 gap-4">
        {group.nodes.map((node) => (
          <SubagentRow key={node.id} node={node} nowMs={nowMs} />
        ))}
      </div>
    </section>
  );
}

function SubagentTree({ tree }: { tree: SubagentNode[] }) {
  const agents = useI18n().t.agents;
  const flat = useMemo(() => flatten(tree), [tree]);
  const groups = useMemo(() => groupDelegations(tree), [tree]);
  const [nowMs, setNowMs] = useState(() => Date.now());

  const active = flat.filter(
    (n) => n.status === "running" || n.status === "queued",
  ).length;
  const failed = flat.filter(
    (n) => n.status === "failed" || n.status === "interrupted",
  ).length;
  const tools = flat.reduce((sum, n) => sum + (n.toolCount ?? 0), 0);
  const files = flat.reduce(
    (sum, n) => sum + n.filesRead.length + n.filesWritten.length,
    0,
  );
  const tokens = flat.reduce(
    (sum, n) => sum + (n.inputTokens ?? 0) + (n.outputTokens ?? 0),
    0,
  );
  const cost = flat.reduce((sum, n) => sum + (n.costUsd ?? 0), 0);

  useEffect(() => {
    if (active <= 0) return;
    const id = window.setInterval(() => setNowMs(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [active]);

  if (tree.length === 0) {
    return (
      <div className="grid place-items-center gap-3 py-12 text-center">
        <Sparkles className="size-6 text-muted-foreground/60" />
        <p className="text-sm font-medium text-foreground/90">
          {agents.emptyTitle}
        </p>
        <p className="max-w-md text-xs leading-relaxed text-muted-foreground/75">
          {agents.emptyDesc}
        </p>
      </div>
    );
  }

  const summary = [
    agents.agentsCount(flat.length),
    active > 0 ? agents.activeCount(active) : "",
    failed > 0 ? agents.failedCount(failed) : "",
    tools > 0 ? agents.toolsCount(tools) : "",
    files > 0 ? agents.filesCount(files) : "",
    tokens > 0 ? fmtTokens(tokens) : "",
    cost > 0 ? `$${cost.toFixed(2)}` : "",
  ].filter(Boolean);

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 overflow-hidden">
      <p className="shrink-0 text-[0.7rem] text-muted-foreground/70">
        {summary.join(" · ")}
      </p>
      <div className="min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain pr-1">
        <div className="flex min-w-0 flex-col gap-6">
          {groups.map((group) => (
            <DelegationGroup group={group} key={group.id} nowMs={nowMs} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Page component ────────────────────────────────────────────────────

export default function AgentsPage() {
  const { t } = useI18n();

  // Subscribe to the module-level subagent store.
  const [, setTick] = useState(0);
  useEffect(() => subscribe(() => setTick((n) => n + 1)), []);

  const subagentsBySession = getSubagentsBySession();

  // Show subagents from all sessions (the desktop filters by active session;
  // the web dashboard has no concept of "active session" — show everything).
  const allSubagents = useMemo(() => {
    const all: SubagentProgress[] = [];
    for (const list of Object.values(subagentsBySession)) {
      all.push(...list);
    }
    return all;
  }, [subagentsBySession]);

  const tree = useMemo(
    () => buildSubagentTree(allSubagents),
    [allSubagents],
  );
  const runningCount = useMemo(
    () => activeSubagentCount(allSubagents),
    [allSubagents],
  );

  return (
    <div className="flex h-full flex-col gap-1 overflow-hidden">
      {/* page header — matches other web pages */}
      <header className="flex shrink-0 items-center gap-3 px-6 pt-6">
        <Cpu className="size-5 text-muted-foreground" />
        <div>
          <h1 className="text-lg font-semibold text-foreground">
            {t.agents.title}
          </h1>
          <p className="text-sm text-muted-foreground">
            {runningCount > 0
              ? t.agents.activeCount(runningCount)
              : t.agents.subtitle}
          </p>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-hidden px-6 pb-6">
        <SubagentTree tree={tree} />
      </div>
    </div>
  );
}
