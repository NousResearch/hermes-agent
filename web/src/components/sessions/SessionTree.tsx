/**
 * SessionTree.tsx
 *
 * Reusable component that renders a flat session list as a hierarchy:
 *   - root (main) sessions at top level
 *   - subagent children indented below their parent
 *   - kanban_worker sessions in a collapsible "Kanban workers (n)" section
 *   - orphan subagents (parent not in list) shown as roots with a marker
 *
 * Row click / toggle / select behaviour is fully delegated via callbacks so
 * the caller (SessionsPage, ProjectsWorkbench) keeps full control.
 *
 * Expand state is persisted to localStorage under the key "sessionTree.expanded".
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Users, Bot, Wrench } from "lucide-react";
import type { SessionInfo } from "@/lib/api";
import { buildSessionTree, type TreeNode } from "@/lib/sessionTree";

// ── Expand-state persistence ─────────────────────────────────────────────────

const LS_KEY = "sessionTree.expanded";

function readExpanded(): Set<string> {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (Array.isArray(parsed)) return new Set(parsed as string[]);
  } catch {
    // ignore
  }
  return new Set();
}

function writeExpanded(set: Set<string>): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify([...set]));
  } catch {
    // ignore — quota exceeded etc.
  }
}

// ── Sub-count badge ───────────────────────────────────────────────────────────

function SubBadge({ count, icon: Icon, label }: { count: number; icon: React.ElementType; label: string }) {
  if (count === 0) return null;
  return (
    <span
      className="inline-flex items-center gap-0.5 rounded-sm bg-midground/10 px-1 py-0.5 font-mono text-[10px] text-midground/60"
      title={label}
    >
      <Icon className="size-2.5" />
      {count}
    </span>
  );
}

// ── Worker row (simple — no expand, no select) ────────────────────────────────

interface WorkerRowProps {
  session: SessionInfo;
  isSelected: boolean;
  onClick: () => void;
}

function WorkerRow({ session, isSelected, onClick }: WorkerRowProps) {
  return (
    <div
      role="button"
      tabIndex={0}
      className={`grid grid-cols-[auto_1fr_auto] gap-2.5 items-center px-2.5 py-2 border-b border-midground/10 cursor-pointer transition-colors hover:bg-midground/5 focus:outline-none ${isSelected ? "bg-primary/[0.06] border-l-2 border-l-primary/40" : ""}`}
      onClick={onClick}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
    >
      <Wrench className="size-3 text-midground/30 shrink-0" />
      <div className="min-w-0">
        <div className="text-sm truncate text-midground/80">
          {session.title || session.id.slice(0, 16)}
        </div>
        {session.kanban_task_id && (
          <div className="font-mono text-[10px] text-midground/40 truncate">
            task {session.kanban_task_id.slice(0, 10)}
          </div>
        )}
      </div>
      {session.is_active && (
        <span className="size-1.5 rounded-full bg-success animate-pulse shrink-0" />
      )}
    </div>
  );
}

// ── Tree node row ─────────────────────────────────────────────────────────────

export interface SessionRowCallbacks {
  onSelectClick: (id: string, event: React.MouseEvent) => void;
  onOpen: (id: string) => void;
  isSelected: (id: string) => boolean;
}

interface TreeNodeRowProps {
  node: TreeNode;
  depth: number;
  callbacks: SessionRowCallbacks;
  isOrphan?: boolean;
  /** Whether the tree node's children (subagents) are expanded */
  subExpanded: boolean;
  onSubToggle: (id: string) => void;
  selectedSessionId?: string;
}

function TreeNodeRow({
  node,
  depth,
  callbacks,
  isOrphan = false,
  subExpanded,
  onSubToggle,
  selectedSessionId,
}: TreeNodeRowProps) {
  const { session, children } = node;
  const hasChildren = children.length > 0;
  const isSelected = callbacks.isSelected(session.id);
  const isActive = !!session.is_active;
  const isHighlighted = selectedSessionId === session.id || children.some((c) => c.session.id === selectedSessionId);

  // Auto-expand parent if selected session is a child
  useEffect(() => {
    if (!selectedSessionId) return;
    if (children.some((c) => c.session.id === selectedSessionId) && !subExpanded) {
      onSubToggle(session.id);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSessionId]);

  return (
    <>
      <div
        className={`flex items-start gap-2 px-2.5 py-2.5 border-b border-midground/10 cursor-pointer transition-colors hover:bg-midground/5 ${
          isSelected
            ? "bg-primary/[0.06] border-l-2 border-primary/40"
            : isActive
              ? "bg-success/[0.03] border-l-2 border-success/30"
              : "border-l-2 border-transparent"
        } ${isHighlighted && !isSelected ? "bg-midground/[0.03]" : ""}`}
        style={{ paddingLeft: `${depth * 16 + 10}px` }}
        onClick={() => callbacks.onOpen(session.id)}
      >
        {/* Sub-tree chevron */}
        {hasChildren ? (
          <button
            className="mt-0.5 shrink-0 text-midground/40 hover:text-midground transition-colors focus:outline-none"
            onClick={(e) => {
              e.stopPropagation();
              onSubToggle(session.id);
            }}
            aria-label={subExpanded ? "Collapse subagents" : "Expand subagents"}
          >
            {subExpanded ? (
              <ChevronDown className="size-3.5" />
            ) : (
              <ChevronRight className="size-3.5" />
            )}
          </button>
        ) : (
          <span className="size-3.5 shrink-0 mt-0.5" />
        )}

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 min-w-0">
            <span
              className={`font-mondwest normal-case min-w-0 flex-1 truncate text-sm ${
                session.title && session.title !== "Untitled"
                  ? "font-medium"
                  : "text-muted-foreground italic"
              }`}
            >
              {session.title && session.title !== "Untitled"
                ? session.title
                : session.preview
                  ? session.preview.slice(0, 60)
                  : "Untitled"}
            </span>
            {isActive && (
              <span className="size-1.5 rounded-full bg-success animate-pulse shrink-0" title="Live" />
            )}
            {isOrphan && (
              <span className="text-[10px] font-mono text-warning/70 shrink-0">orphan</span>
            )}
          </div>

          <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
            {session.model && (
              <span className="font-mono text-[11px] text-midground/40 truncate max-w-[120px]">
                {session.model.split("/").pop()}
              </span>
            )}
            {hasChildren && (
              <SubBadge
                count={children.length}
                icon={Bot}
                label={`${children.length} subagent${children.length !== 1 ? "s" : ""}`}
              />
            )}
          </div>
        </div>
      </div>

      {/* Children (subagents) */}
      {hasChildren && subExpanded && (
        <div>
          {children.map((child) => (
            <TreeNodeRow
              key={child.session.id}
              node={child}
              depth={depth + 1}
              callbacks={callbacks}
              subExpanded={false}
              onSubToggle={onSubToggle}
              selectedSessionId={selectedSessionId}
            />
          ))}
        </div>
      )}
    </>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────

export interface SessionTreeProps {
  sessions: SessionInfo[];
  /** Currently selected / open session id (for highlight) */
  selectedSessionId?: string;
  callbacks: SessionRowCallbacks;
  /** Show kanban-worker section. Default true. */
  showWorkers?: boolean;
  className?: string;
}

export function SessionTree({
  sessions,
  selectedSessionId,
  callbacks,
  showWorkers = true,
  className = "",
}: SessionTreeProps) {
  const { roots, workers, orphans } = useMemo(
    () => buildSessionTree(sessions),
    [sessions],
  );

  // Sub-tree expand state (which parent ids have their children visible)
  const [subExpanded, setSubExpanded] = useState<Set<string>>(readExpanded);
  const [workersOpen, setWorkersOpen] = useState(false);

  // Sync to localStorage
  const subExpandedRef = useRef(subExpanded);
  subExpandedRef.current = subExpanded;
  useEffect(() => {
    writeExpanded(subExpanded);
  }, [subExpanded]);

  const onSubToggle = useCallback((id: string) => {
    setSubExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  if (sessions.length === 0) return null;

  return (
    <div className={`flex flex-col min-w-0 ${className}`}>
      {/* Root sessions */}
      {roots.map((node) => (
        <TreeNodeRow
          key={node.session.id}
          node={node}
          depth={0}
          callbacks={callbacks}
          subExpanded={subExpanded.has(node.session.id)}
          onSubToggle={onSubToggle}
          selectedSessionId={selectedSessionId}
        />
      ))}

      {/* Orphans */}
      {orphans.map((s) => (
        <TreeNodeRow
          key={s.id}
          node={{ session: s, children: [] }}
          depth={0}
          callbacks={callbacks}
          isOrphan
          subExpanded={false}
          onSubToggle={onSubToggle}
          selectedSessionId={selectedSessionId}
        />
      ))}

      {/* Kanban workers section */}
      {showWorkers && workers.length > 0 && (
        <div className="mt-1 border-t border-midground/10">
          <button
            className="flex w-full items-center gap-1.5 px-2.5 py-2 text-left hover:bg-midground/5 transition-colors focus:outline-none"
            onClick={() => setWorkersOpen((o) => !o)}
          >
            {workersOpen ? (
              <ChevronDown className="size-3.5 text-midground/40" />
            ) : (
              <ChevronRight className="size-3.5 text-midground/40" />
            )}
            <Users className="size-3.5 text-midground/40" />
            <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/50">
              Kanban workers ({workers.length})
            </span>
          </button>

          {workersOpen && (
            <div>
              {workers.map((s) => (
                <WorkerRow
                  key={s.id}
                  session={s}
                  isSelected={callbacks.isSelected(s.id)}
                  onClick={() => callbacks.onOpen(s.id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
