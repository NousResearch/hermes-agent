/**
 * SubagentTree — Real-time delegation tree from nanostore.
 *
 * Subscribes to subagentStore (populated by gateway WebSocket events)
 * and renders a depth-indented tree via buildSubagentTree.
 *
 * For expanded children, it still uses REST to load past messages
 * (only for completed/failed children, not active ones).
 */

import {
  ChevronRight,
  ChevronDown,
  GitBranch,
  Loader2,
  FileText,
  ExternalLink,
  CircleDollarSign,
  FileCode2,
  MessageSquare,
  Brain,
  Wrench,
  Square,
  History,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "@nanostores/react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { subagentStore } from "@/lib/subagentStore";
import { GatewayClient } from "@/lib/gatewayClient";
import {
  buildSubagentTree,
  treeTotals,
  fmtCost,
  fmtTokens,
  fmtDuration,
  type SubagentNode,
} from "@/lib/subagentTree";
import type { SubagentProgress } from "@/lib/subagentStore";

/* ------------------------------------------------------------------ */
/*  Expanded child state (REST-loaded messages)                      */
/* ------------------------------------------------------------------ */

interface ExpandedChildState {
  sessionId: string;
  messages: Array<{ content: string; timestamp?: number }>;
  loading: boolean;
  error: string | null;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const MAX_MESSAGES_TO_LOAD = 5;
const MESSAGE_TRUNCATE = 200;

function truncate(text: string, max: number): string {
  if (!text) return "";
  if (text.length <= max) return text;
  return text.slice(0, max) + "\u2026";
}

function isRunning(item: SubagentProgress): boolean {
  return item.status === "running" || item.status === "queued";
}

function statusBadge(status: SubagentProgress["status"]): {
  label: string;
  color: string;
  icon: React.ReactNode;
} {
  switch (status) {
    case "running":
      return {
        label: "running",
        color: "text-primary",
        icon: (
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
          </span>
        ),
      };
    case "queued":
      return {
        label: "queued",
        color: "text-text-tertiary",
        icon: <Loader2 className="h-2.5 w-2.5 animate-spin" />,
      };
    case "completed":
      return {
        label: "done",
        color: "text-primary/70",
        icon: <span className="h-2 w-2 rounded-full bg-primary/70" />,
      };
    case "error":
    case "failed":
    case "interrupted":
    case "timeout":
      return {
        label: status,
        color: "text-destructive",
        icon: <span className="h-2 w-2 rounded-full bg-destructive" />,
      };
    default:
      return {
        label: status,
        color: "text-text-tertiary",
        icon: <span className="h-2 w-2 rounded-full bg-text-tertiary/50" />,
      };
  }
}

/* ------------------------------------------------------------------ */
/*  Render a single tree node (recursive)                              */
/* ------------------------------------------------------------------ */

function TreeNode({
  node,
  depth,
  expandedState,
  onExpandChild,
  onOpenSession,
  onInterrupt,
}: {
  node: SubagentNode;
  depth: number;
  expandedState: Record<string, ExpandedChildState>;
  onExpandChild: (id: string) => void;
  onOpenSession: (id: string) => void;
  onInterrupt: (id: string) => void;
}) {
  const { item, children, aggregate } = node;
  const isExpanded = !!expandedState[item.id];
  const childState = expandedState[item.id];
  const running = isRunning(item);

  const badge = statusBadge(item.status);

  return (
    <div className="relative">
      {/* Node row */}
      <div
        className={cn(
          "group flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] transition-colors",
          "hover:bg-muted/20",
        )}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
      >
        {/* Expand/collapse children */}
        {children.length > 0 && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onExpandChild(item.id);
            }}
            className="shrink-0 text-text-tertiary transition-transform hover:text-foreground"
          >
            {isExpanded ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </button>
        )}
        {children.length === 0 && <span className="w-4 shrink-0" />}

        {/* Status dot */}
        <div className="shrink-0">{badge.icon}</div>

        {/* Interrupt button (running/queued only, visible on hover) */}
        {running && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onInterrupt(item.id);
            }}
            className="shrink-0 rounded-full bg-transparent p-0.5 text-destructive opacity-0 transition-opacity group-hover:opacity-100 hover:bg-destructive/20"
            title="Interrupt subagent"
          >
            <Square className="h-3.5 w-3.5" />
          </button>
        )}

        {/* Goal / title */}
        <span className="min-w-0 flex-1 truncate font-mono text-foreground/90">
          {truncate(item.goal, 60)}
        </span>

        {/* Task progress */}
        {item.taskCount > 0 && (
          <span className="shrink-0 tabular-nums text-text-tertiary">
            task {item.index + 1}/{item.taskCount}
          </span>
        )}

        {/* Current tool badge */}
        {running && item.tools.length > 0 && (
          <span className="shrink-0 rounded bg-primary/10 px-1.5 py-px text-[10px] font-medium text-primary">
            <Wrench className="mr-0.5 inline h-2.5 w-2.5" />
            {truncate(item.tools[item.tools.length - 1], 24)}
          </span>
        )}

        {/* Cost */}
        {item.costUsd !== undefined && item.costUsd > 0 && (
          <span className="shrink-0 tabular-nums text-text-tertiary">
            <CircleDollarSign className="mr-0.5 inline h-2.5 w-2.5" />
            {fmtCost(item.costUsd)}
          </span>
        )}

        {/* Files touched */}
        {aggregate.filesTouched > 0 && (
          <span className="shrink-0 tabular-nums text-text-tertiary">
            <FileCode2 className="mr-0.5 inline h-2.5 w-2.5" />
            {aggregate.filesTouched}
          </span>
        )}

        {/* Tokens */}
        {(item.inputTokens ?? 0) + (item.outputTokens ?? 0) > 0 && (
          <span className="shrink-0 tabular-nums text-text-tertiary">
            {fmtTokens(item.inputTokens! + item.outputTokens!)} tok
          </span>
        )}

        {/* Duration */}
        {item.durationSeconds !== undefined && item.durationSeconds! > 0 && (
          <span className="shrink-0 tabular-nums text-text-tertiary">
            {fmtDuration(item.durationSeconds!)}
          </span>
        )}

        {/* Depth badge */}
        {depth > 0 && (
          <span className="shrink-0 rounded bg-muted/40 px-1 py-px text-[9px] text-text-tertiary/60">
            d{depth}
          </span>
        )}
      </div>

      {/* Expanded child detail (REST messages) */}
      {isExpanded && childState && (
        <div className="ml-5 mt-1 space-y-2 rounded-md border border-border/30 bg-muted/5 p-2.5">
          {/* Real-time streams: tools, thinking, notes */}
          {(item.tools.length > 0 || item.thinking.length > 0 || item.notes.length > 0) && (
            <div className="space-y-1.5">
              {/* Tools stream */}
              {item.tools.length > 0 && (
                <div>
                  <div className="mb-1 flex items-center gap-1 text-[10px] font-medium text-text-tertiary">
                    <Wrench className="h-2.5 w-2.5" />
                    Tools ({item.tools.length})
                  </div>
                  <div className="space-y-0.5">
                    {item.tools.map((t, i) => (
                      <div
                        key={i}
                        className="rounded bg-background/40 px-2 py-0.5 text-[10px] font-mono text-text-secondary"
                      >
                        {truncate(t, 80)}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Thinking stream */}
              {item.thinking.length > 0 && (
                <div>
                  <div className="mb-1 flex items-center gap-1 text-[10px] font-medium text-text-tertiary">
                    <Brain className="h-2.5 w-2.5" />
                    Thinking ({item.thinking.length})
                  </div>
                  <div className="space-y-0.5">
                    {item.thinking.map((t, i) => (
                      <div
                        key={i}
                        className="rounded bg-background/40 px-2 py-0.5 text-[10px] leading-relaxed text-text-secondary"
                      >
                        {truncate(t, 120)}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Notes stream */}
              {item.notes.length > 0 && (
                <div>
                  <div className="mb-1 flex items-center gap-1 text-[10px] font-medium text-text-tertiary">
                    <MessageSquare className="h-2.5 w-2.5" />
                    Notes ({item.notes.length})
                  </div>
                  <div className="space-y-0.5">
                    {item.notes.map((n, i) => (
                      <div
                        key={i}
                        className="rounded bg-background/40 px-2 py-0.5 text-[10px] leading-relaxed text-text-secondary"
                      >
                        {truncate(n, 120)}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Aggregate summary */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-text-tertiary">
            {aggregate.totalTools > 0 && (
              <span>
                {aggregate.totalTools} tools
              </span>
            )}
            {aggregate.totalDuration > 0 && (
              <span>{fmtDuration(aggregate.totalDuration)}</span>
            )}
            {aggregate.costUsd > 0 && (
              <span>{fmtCost(aggregate.costUsd)}</span>
            )}
            {aggregate.filesTouched > 0 && (
              <span>
                {aggregate.filesTouched} files
              </span>
            )}
            {aggregate.activeCount > 0 && (
              <span className="text-primary">
                ⚡ {aggregate.activeCount} active
              </span>
            )}
            {aggregate.descendantCount > 0 && (
              <span>{aggregate.descendantCount} descendants</span>
            )}
          </div>

          {/* REST-loaded messages (for completed/failed children) */}
          {!running && (
            <div>
              {childState.loading ? (
                <div className="flex items-center gap-2 text-xs text-text-tertiary">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Loading messages…
                </div>
              ) : childState.error ? (
                <div className="text-xs text-destructive/80">
                  Failed to load: {childState.error}
                </div>
              ) : childState.messages.length === 0 ? (
                <div className="text-xs text-text-tertiary/60">
                  No assistant messages found.
                </div>
              ) : (
                <div className="space-y-2">
                  {childState.messages.map((msg, mi) => (
                    <div
                      key={mi}
                      className="flex items-start gap-1.5 rounded bg-background/40 px-2 py-1.5"
                    >
                      <FileText className="mt-0.5 h-2.5 w-2.5 shrink-0 text-text-tertiary/50" />
                      <span className="text-[11px] leading-relaxed text-text-secondary">
                        {truncate(msg.content, MESSAGE_TRUNCATE)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Open in Sessions */}
          <div className="flex items-center justify-end">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onOpenSession(item.id);
              }}
              className="flex items-center gap-1 text-[10px] font-medium text-primary hover:text-primary/80 transition-colors"
            >
              Open in Sessions
              <ExternalLink className="h-2.5 w-2.5" />
            </button>
          </div>
        </div>
      )}

      {/* Recursive children */}
      {isExpanded &&
       children.map((child) => (
         <TreeNode
           key={child.item.id}
           node={child}
           depth={depth + 1}
           expandedState={expandedState}
           onExpandChild={onExpandChild}
           onOpenSession={onOpenSession}
           onInterrupt={onInterrupt}
         />
       ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function SubagentTree({ sessionId }: { sessionId: string }) {
  // Gateway client for spawn tree operations (separate from the interrupt client)
  const [spawnGw] = useState(() => {
    const client = new GatewayClient();
    client.connect().catch(() => {});
    return client;
  });
  useEffect(() => {
    return () => spawnGw.close();
  }, [spawnGw]);

  // History panel state
  const [showHistory, setShowHistory] = useState(false);
  const [treesList, setTreesList] = useState<
    Array<{
      path: string;
      session_id: string;
      started_at: number;
      finished_at: number;
      label: string;
      count: number;
    }>
  >([]);
  const [loadedTree, setLoadedTree] = useState<{
    session_id: string;
    started_at: number;
    finished_at: number;
    label: string;
    subagents: object[];
  } | null>(null);

  // Gateway client for interrupt RPC
  const [gw] = useState(() => {
    const client = new GatewayClient();
    client.connect().catch(() => {});
    return client;
  });

  // Cleanup gateway client on unmount
  useEffect(() => {
    return () => gw.close();
  }, [gw]);

  // Real-time subagents from nanostore
  const subagents = useStore(subagentStore);

  // Build tree from real-time data
  const tree = useMemo(
    () => buildSubagentTree(Object.values(subagents)),
    [subagents],
  );

  // Totals for header
  const totals = useMemo(() => treeTotals(tree), [tree]);

  const [expanded, setExpanded] = useState(false);
  const [expandedState, setExpandedState] = useState<Record<string, ExpandedChildState>>({});
  const hasAutoExpandedRef = useRef(false);
  const firstLoadRef = useRef(true);
  const navigate = useNavigate();

  const expandChild = useCallback(
    async (childId: string) => {
      // If already expanded, collapse it
      if (expandedState[childId]) {
        const next = { ...expandedState };
        delete next[childId];
        setExpandedState(next);
        return;
      }

      setExpandedState((prev) => ({
        ...prev,
        [childId]: { sessionId: childId, messages: [], loading: true, error: null },
      }));

      try {
        const res = await api.getSessionMessages(childId);
        const assistantMsgs = res.messages
          .filter((m) => m.role === "assistant")
          .slice(-MAX_MESSAGES_TO_LOAD);
        const msgs = assistantMsgs.map((m) => ({
          content: m.content ?? "",
          timestamp: m.timestamp,
        }));

        setExpandedState((prev) => ({
          ...prev,
          [childId]: { sessionId: childId, messages: msgs, loading: false, error: null },
        }));
      } catch (err) {
        setExpandedState((prev) => ({
          ...prev,
          [childId]: {
            sessionId: childId,
            messages: [],
            loading: false,
            error: err instanceof Error ? err.message : "Failed to load messages",
          },
        }));
      }
    },
    [expandedState],
  );

  const handleOpenSession = useCallback(
    (id: string) => {
      navigate(`/sessions?resume=${id}`);
    },
    [navigate],
  );

  // Interrupt a subagent with optimistic UI update
  const handleInterrupt = useCallback(
    async (id: string) => {
      // Optimistic: set status to 'interrupted' immediately
      const store = subagentStore.get();
      const existing = store[id];
      if (existing) {
        subagentStore.set({
          ...store,
          [id]: { ...existing, status: "interrupted" },
        });
      }
      // Call the RPC (non-blocking — fire and forget)
      gw.interruptSubagent(id).catch(() => {
        // If the RPC fails, the gateway will eventually send the real
        // subagent.interrupt event which will update the store correctly.
      });
    },
    [gw],
  );

  // Relative time formatter
  function relativeTime(ts: number): string {
    const diff = Date.now() - ts;
    const mins = Math.floor(diff / 60_000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ago`;
  }

  // Open history panel (load list of saved trees)
  const openHistory = useCallback(async () => {
    try {
      const res = await spawnGw.listSpawnTrees(sessionId);
      setTreesList(res.entries ?? []);
      setShowHistory(true);
    } catch {
      setTreesList([]);
      setShowHistory(true);
    }
  }, [spawnGw, sessionId]);

  // Close history panel, return to live view
  const closeHistory = useCallback(() => {
    setShowHistory(false);
    setTreesList([]);
    setLoadedTree(null);
  }, []);

  // Load a specific spawn tree for replay
  const loadSpawnTree = useCallback(
    async (entry: { path: string; label: string }) => {
      try {
        const res = await spawnGw.loadSpawnTree(entry.path);
        setLoadedTree(res);
      } catch {
        setLoadedTree(null);
      }
    },
    [spawnGw],
  );

  // Auto-expand when running children first appear (no polling needed — nanostore updates trigger re-render)
  useEffect(() => {
    if (firstLoadRef.current) {
      firstLoadRef.current = false;
    }
    if (!hasAutoExpandedRef.current && totals.activeCount > 0) {
      setExpanded(true);
      hasAutoExpandedRef.current = true;
    }
  }, [totals.activeCount]);

  const hasChildren = Object.keys(subagents).length > 0;

  return (
    <div
      className={cn(
        "border-b border-border/40 transition-colors",
        hasChildren ? "bg-muted/15" : "bg-muted/10",
      )}
    >
      {/* Header bar */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className={cn(
          "flex w-full items-center gap-1.5 px-3 py-1.5 text-xs transition-colors",
          "hover:text-foreground hover:bg-muted/20",
          "text-text-secondary",
        )}
      >
        <GitBranch
          className={cn(
            "h-3.5 w-3.5 shrink-0 transition-colors",
            hasChildren && "text-primary/80",
          )}
        />
        <span className="font-mondwest tracking-wider">subagents</span>

        {/* Total child count badge */}
        <span
          className={cn(
            "inline-flex min-w-4 items-center justify-center rounded-full px-1.5 text-[10px] font-medium",
            hasChildren
              ? "bg-primary/15 text-primary"
              : "bg-muted/60 text-text-tertiary",
          )}
        >
          {totals.descendantCount + totals.activeCount}
        </span>

        {/* Active children indicator */}
        {totals.activeCount > 0 && (
          <span className="inline-flex items-center gap-0.5 text-[10px] text-primary">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-primary" />
            </span>
            {totals.activeCount} active
          </span>
        )}

        {/* History button — only when no active subagents */}
        {totals.activeCount === 0 && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              openHistory();
            }}
            className="ml-auto flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-text-tertiary transition-colors hover:bg-muted/30 hover:text-foreground"
            title="View saved spawn trees"
          >
            <History className="h-3 w-3" />
            <span>History</span>
          </button>
        )}

        {/* Chevron toggle */}
        <ChevronRight
          className={cn(
            "ml-auto h-3 w-3 shrink-0 transition-transform",
            expanded && "rotate-90",
          )}
        />
      </button>

      {/* History dropdown / replay panel */}
      {showHistory && (
        <div className="border-t border-border/40 px-3 py-2">
          {/* History header */}
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium text-text-secondary">
              {loadedTree ? `History: ${loadedTree.label}` : "Saved spawn trees"}
            </span>
            <button
              onClick={closeHistory}
              className="rounded p-0.5 text-text-tertiary transition-colors hover:bg-muted/30 hover:text-foreground"
              title="Close history"
            >
              <span className="text-xs">×</span>
            </button>
          </div>

          {/* Trees list (when no tree is loaded) */}
          {!loadedTree && (
            <div className="space-y-1">
              {treesList.length === 0 ? (
                <div className="flex flex-col items-center gap-1 py-4 text-center">
                  <History className="h-4 w-4 text-text-tertiary/40" />
                  <p className="text-xs text-text-tertiary">
                    No saved spawn trees for this session.
                  </p>
                </div>
              ) : (
                treesList.map((entry) => (
                  <button
                    key={entry.path}
                    onClick={() => loadSpawnTree(entry)}
                    className="flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-muted/20"
                  >
                    <span className="min-w-0 flex-1 truncate font-mono text-foreground/80">
                      {entry.label}
                    </span>
                    <span className="ml-2 shrink-0 tabular-nums text-text-tertiary">
                      {entry.count} agent{entry.count !== 1 ? "s" : ""} · {relativeTime(entry.started_at)}
                    </span>
                  </button>
                ))
              )}
            </div>
          )}

          {/* Replay view (when a tree is loaded) */}
          {loadedTree && (
            <div>
              {/* Replay tree nodes */}
              {(() => {
                const replaySubagents = loadedTree.subagents as unknown as SubagentProgress[];
                const replayTree = buildSubagentTree(replaySubagents);
                return (
                  <>
                    {replayTree.length === 0 ? (
                      <div className="py-4 text-center text-xs text-text-tertiary">
                        No subagents in this saved tree.
                      </div>
                    ) : (
                      replayTree.map((node) => (
                        <TreeNode
                          key={node.item.id}
                          node={node}
                          depth={0}
                          expandedState={expandedState}
                          onExpandChild={expandChild}
                          onOpenSession={handleOpenSession}
                          onInterrupt={handleInterrupt}
                        />
                      ))
                    )}
                  </>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* Expanded panel */}
      {expanded && (
        <div className="border-t border-border/40 px-3 py-2">
          {/* Empty state */}
          {!hasChildren && (
            <div className="flex flex-col items-center gap-1 py-4 text-center">
              <GitBranch className="h-4 w-4 text-text-tertiary/40" />
              <p className="text-xs text-text-tertiary">
                No active delegation trees.
              </p>
              <p className="text-[10px] text-text-tertiary/60">
                When delegate_task is used, child sessions appear here in real-time.
              </p>
            </div>
          )}

          {/* Render tree nodes */}
          {tree.map((node) => (
            <TreeNode
              key={node.item.id}
              node={node}
              depth={0}
              expandedState={expandedState}
              onExpandChild={expandChild}
              onOpenSession={handleOpenSession}
              onInterrupt={handleInterrupt}
            />
          ))}
        </div>
      )}
    </div>
  );
}
