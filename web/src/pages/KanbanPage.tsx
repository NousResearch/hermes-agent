import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileDiff,
  Loader2,
  Play,
  RefreshCw,
  Square,
  Terminal,
  Trash2,
} from "lucide-react";
import { H2 } from "@nous-research/ui";
import { api } from "@/lib/api";
import type {
  KanbanCard,
  KanbanCardDiffResponse,
  KanbanCardLogResponse,
  KanbanColumn,
  KanbanColumnId,
  KanbanTaskStatus,
} from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const STATUS_LABEL: Record<KanbanTaskStatus, string> = {
  idle: "Idle",
  running: "Running",
  review: "Review",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

const STATUS_STYLE: Record<KanbanTaskStatus, string> = {
  idle: "border-muted-foreground/30 text-muted-foreground",
  running: "border-warning/40 bg-warning/10 text-warning",
  review: "border-foreground/30 bg-foreground/10 text-foreground",
  done: "border-success/40 bg-success/10 text-success",
  failed: "border-destructive/40 bg-destructive/10 text-destructive",
  stopped: "border-muted-foreground/30 bg-muted text-muted-foreground",
};

function byUpdatedAt(a: KanbanCard, b: KanbanCard) {
  return b.updated_at - a.updated_at;
}

function StatusBadge({ status }: { status: KanbanTaskStatus }) {
  return (
    <span
      className={cn(
        "inline-flex items-center border px-2 py-0.5 font-compressed text-[0.62rem] tracking-[0.14em] uppercase",
        STATUS_STYLE[status],
      )}
    >
      {status === "running" && <Loader2 className="mr-1 h-3 w-3 animate-spin" />}
      {status === "done" && <CheckCircle2 className="mr-1 h-3 w-3" />}
      {status === "failed" && <AlertCircle className="mr-1 h-3 w-3" />}
      {STATUS_LABEL[status]}
    </span>
  );
}

function BoardCard({
  card,
  selected,
  onSelect,
  onStart,
  onStop,
}: {
  card: KanbanCard;
  selected: boolean;
  onSelect: () => void;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "group w-full border bg-card/70 p-3 text-left transition-colors",
        "hover:bg-card focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        selected ? "border-foreground/55" : "border-border",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <h3 className="min-w-0 text-sm font-semibold normal-case tracking-normal text-foreground">
          {card.title}
        </h3>
        <StatusBadge status={card.status} />
      </div>
      <p className="mt-2 line-clamp-3 text-xs normal-case leading-relaxed tracking-normal text-muted-foreground">
        {card.prompt}
      </p>
      <div className="mt-3 flex items-center justify-between gap-2 text-[0.65rem] text-muted-foreground">
        <span className="min-w-0 truncate font-mono-ui">
          {card.workspace_path ?? "workspace unset"}
        </span>
        <span className="shrink-0">{timeAgo(card.updated_at)}</span>
      </div>
      {card.last_activity && (
        <div className="mt-2 border-t border-border pt-2 text-[0.68rem] normal-case leading-snug tracking-normal text-muted-foreground/90">
          {card.last_activity}
        </div>
      )}
      <div className="mt-3 flex items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
        {card.status === "running" ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-7"
            onClick={(event) => {
              event.stopPropagation();
              onStop();
            }}
          >
            <Square className="h-3 w-3" />
            Stop
          </Button>
        ) : (
          <Button
            type="button"
            size="sm"
            className="h-7"
            onClick={(event) => {
              event.stopPropagation();
              onStart();
            }}
          >
            <Play className="h-3 w-3" />
            Start
          </Button>
        )}
      </div>
    </button>
  );
}

function DetailPanel({
  card,
  diff,
  log,
  busy,
  onRefresh,
  onStart,
  onStop,
  onSave,
  onMove,
  onDelete,
}: {
  card: KanbanCard | null;
  diff: KanbanCardDiffResponse | null;
  log: KanbanCardLogResponse | null;
  busy: boolean;
  onRefresh: () => void;
  onStart: () => void;
  onStop: () => void;
  onSave: (values: { title: string; prompt: string; workspace_path: string }) => void;
  onMove: (column: KanbanColumnId) => void;
  onDelete: () => void;
}) {
  const [editTitle, setEditTitle] = useState("");
  const [editPrompt, setEditPrompt] = useState("");
  const [editWorkspace, setEditWorkspace] = useState("");

  useEffect(() => {
    setEditTitle(card?.title ?? "");
    setEditPrompt(card?.prompt ?? "");
    setEditWorkspace(card?.workspace_path ?? "");
  }, [card?.id, card?.prompt, card?.title, card?.workspace_path]);

  if (!card) {
    return (
      <Card className="lg:sticky lg:top-20">
        <CardHeader>
          <CardTitle>Task</CardTitle>
        </CardHeader>
        <CardContent className="text-sm normal-case tracking-normal text-muted-foreground">
          Select a card.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="lg:sticky lg:top-20">
      <CardHeader className="gap-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <CardTitle className="truncate">{card.title}</CardTitle>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <StatusBadge status={card.status} />
              {card.pid && <Badge variant="outline">PID {card.pid}</Badge>}
            </div>
          </div>
          <Button size="icon" variant="ghost" onClick={onRefresh} disabled={busy} aria-label="Refresh card">
            <RefreshCw className={cn("h-4 w-4", busy && "animate-spin")} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="mb-1 font-compressed text-[0.7rem] uppercase tracking-[0.16em] text-muted-foreground">
            Edit Card
          </div>
          <Input
            value={editTitle}
            onChange={(event) => setEditTitle(event.target.value)}
            disabled={card.status === "running"}
            className="normal-case tracking-normal"
          />
          <textarea
            value={editPrompt}
            onChange={(event) => setEditPrompt(event.target.value)}
            disabled={card.status === "running"}
            className={cn(
              "min-h-28 w-full resize-y border border-input bg-transparent px-3 py-2",
              "text-sm normal-case tracking-normal text-foreground",
              "disabled:cursor-not-allowed disabled:opacity-50",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            )}
          />
          <Input
            value={editWorkspace}
            onChange={(event) => setEditWorkspace(event.target.value)}
            disabled={card.status === "running"}
            className="font-mono-ui text-xs normal-case tracking-normal"
          />
          <Button
            variant="outline"
            disabled={
              card.status === "running" ||
              !editTitle.trim() ||
              !editPrompt.trim() ||
              !editWorkspace.trim()
            }
            onClick={() =>
              onSave({
                title: editTitle,
                prompt: editPrompt,
                workspace_path: editWorkspace,
              })
            }
          >
            Save
          </Button>
        </div>

        <div className="grid grid-cols-2 gap-2">
          {card.status === "running" ? (
            <Button variant="outline" onClick={onStop}>
              <Square className="h-4 w-4" />
              Stop
            </Button>
          ) : (
            <Button onClick={onStart}>
              <Play className="h-4 w-4" />
              Start
            </Button>
          )}
          <Button variant="outline" onClick={() => onMove("done")}>
            <CheckCircle2 className="h-4 w-4" />
            Done
          </Button>
          <Button variant="outline" onClick={() => onMove("backlog")}>
            Backlog
          </Button>
          <Button variant="outline" onClick={() => onMove("trash")}>
            Trash
          </Button>
          <Button variant="destructive" onClick={onDelete}>
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </div>

        <section>
          <div className="mb-2 flex items-center gap-2 font-compressed text-[0.7rem] uppercase tracking-[0.16em] text-muted-foreground">
            <Terminal className="h-3.5 w-3.5" />
            Log
          </div>
          <pre className="max-h-52 overflow-auto border border-border bg-black/35 p-3 text-[0.7rem] normal-case leading-relaxed tracking-normal text-muted-foreground">
            {log?.log || "No log yet."}
          </pre>
        </section>

        <section>
          <div className="mb-2 flex items-center gap-2 font-compressed text-[0.7rem] uppercase tracking-[0.16em] text-muted-foreground">
            <FileDiff className="h-3.5 w-3.5" />
            Diff
          </div>
          <div className="mb-2 border border-border bg-background/40 px-2 py-1 text-xs normal-case tracking-normal text-muted-foreground">
            {diff?.summary ?? "No diff loaded."}
          </div>
          <pre className="max-h-64 overflow-auto border border-border bg-black/35 p-3 text-[0.7rem] normal-case leading-relaxed tracking-normal text-muted-foreground">
            {diff?.diff || "No patch."}
          </pre>
        </section>
      </CardContent>
    </Card>
  );
}

export default function KanbanPage() {
  const [columns, setColumns] = useState<KanbanColumn[]>([]);
  const [cards, setCards] = useState<KanbanCard[]>([]);
  const [defaultWorkspace, setDefaultWorkspace] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [workspace, setWorkspace] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState<KanbanCardLogResponse | null>(null);
  const [diff, setDiff] = useState<KanbanCardDiffResponse | null>(null);

  const selectedCard = useMemo(
    () => cards.find((card) => card.id === selectedId) ?? null,
    [cards, selectedId],
  );

  const groupedCards = useMemo(() => {
    const groups = new Map<KanbanColumnId, KanbanCard[]>();
    for (const column of columns) groups.set(column.id, []);
    for (const card of cards) {
      groups.get(card.column)?.push(card);
    }
    for (const group of groups.values()) group.sort(byUpdatedAt);
    return groups;
  }, [cards, columns]);

  const refreshBoard = useCallback(async () => {
    const response = await api.getKanbanBoard();
    setColumns(response.columns);
    setCards(response.board.cards);
    setDefaultWorkspace(response.default_workspace_path);
    setWorkspace((current) => current || response.default_workspace_path);
    if (!selectedId && response.board.cards.length > 0) {
      setSelectedId(response.board.cards[0].id);
    }
  }, [selectedId]);

  const refreshDetail = useCallback(async (cardId: string) => {
    const [nextLog, nextDiff] = await Promise.all([
      api.getKanbanCardLog(cardId),
      api.getKanbanCardDiff(cardId),
    ]);
    setLog(nextLog);
    setDiff(nextDiff);
  }, []);

  const runAction = useCallback(async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    runAction(refreshBoard);
  }, [refreshBoard, runAction]);

  useEffect(() => {
    if (!selectedCard) {
      setLog(null);
      setDiff(null);
      return;
    }
    runAction(() => refreshDetail(selectedCard.id));
  }, [selectedCard?.id, refreshDetail, runAction]);

  useEffect(() => {
    if (!cards.some((card) => card.status === "running")) return;
    const id = window.setInterval(() => {
      refreshBoard().catch(() => {});
      if (selectedId) refreshDetail(selectedId).catch(() => {});
    }, 4000);
    return () => window.clearInterval(id);
  }, [cards, refreshBoard, refreshDetail, selectedId]);

  function replaceCard(card: KanbanCard) {
    setCards((current) => current.map((item) => (item.id === card.id ? card : item)));
    setSelectedId(card.id);
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <H2 className="blend-lighter">Kanban</H2>
          <div className="mt-1 text-xs normal-case tracking-normal text-muted-foreground">
            {cards.length} cards · {cards.filter((card) => card.status === "running").length} running
          </div>
        </div>
        <Button variant="outline" onClick={() => runAction(refreshBoard)} disabled={busy}>
          <RefreshCw className={cn("h-4 w-4", busy && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm normal-case tracking-normal text-destructive">
          {error}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>New Card</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[1fr_1.6fr_1fr_auto]">
          <Input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder="Title"
            className="normal-case tracking-normal"
          />
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Prompt"
            className={cn(
              "min-h-9 resize-y border border-input bg-transparent px-3 py-2",
              "text-sm normal-case tracking-normal text-foreground placeholder:text-muted-foreground",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            )}
          />
          <Input
            value={workspace}
            onChange={(event) => setWorkspace(event.target.value)}
            placeholder={defaultWorkspace || "Workspace path"}
            className="font-mono-ui text-xs normal-case tracking-normal"
          />
          <Button
            disabled={busy || !title.trim() || !prompt.trim()}
            onClick={() =>
              runAction(async () => {
                const card = await api.createKanbanCard({
                  title,
                  prompt,
                  workspace_path: workspace || defaultWorkspace,
                });
                setCards((current) => [card, ...current]);
                setSelectedId(card.id);
                setTitle("");
                setPrompt("");
              })
            }
          >
            Add
          </Button>
        </CardContent>
      </Card>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          {columns.map((column) => {
            const items = groupedCards.get(column.id) ?? [];
            return (
              <section key={column.id} className="min-w-0 border border-border bg-background/25">
                <div className="flex items-center justify-between border-b border-border px-3 py-2">
                  <h2 className="font-expanded text-xs font-bold uppercase tracking-[0.12em] text-foreground">
                    {column.title}
                  </h2>
                  <Badge variant="outline">{items.length}</Badge>
                </div>
                <div className="flex min-h-72 flex-col gap-2 p-2">
                  {items.map((card) => (
                    <BoardCard
                      key={card.id}
                      card={card}
                      selected={card.id === selectedId}
                      onSelect={() => setSelectedId(card.id)}
                      onStart={() =>
                        runAction(async () => {
                          const updated = await api.startKanbanCard(card.id, card.workspace_path ?? workspace);
                          replaceCard(updated);
                          await refreshDetail(updated.id);
                        })
                      }
                      onStop={() =>
                        runAction(async () => {
                          const updated = await api.stopKanbanCard(card.id);
                          replaceCard(updated);
                          await refreshDetail(updated.id);
                        })
                      }
                    />
                  ))}
                </div>
              </section>
            );
          })}
        </div>

        <DetailPanel
          card={selectedCard}
          log={log}
          diff={diff}
          busy={busy}
          onRefresh={() =>
            selectedCard &&
            runAction(async () => {
              await refreshBoard();
              await refreshDetail(selectedCard.id);
            })
          }
          onStart={() =>
            selectedCard &&
            runAction(async () => {
              const updated = await api.startKanbanCard(
                selectedCard.id,
                selectedCard.workspace_path ?? workspace,
              );
              replaceCard(updated);
              await refreshDetail(updated.id);
            })
          }
          onStop={() =>
            selectedCard &&
            runAction(async () => {
              const updated = await api.stopKanbanCard(selectedCard.id);
              replaceCard(updated);
              await refreshDetail(updated.id);
            })
          }
          onSave={(values) =>
            selectedCard &&
            runAction(async () => {
              const updated = await api.updateKanbanCard(selectedCard.id, values);
              replaceCard(updated);
              await refreshDetail(updated.id);
            })
          }
          onMove={(column) =>
            selectedCard &&
            runAction(async () => {
              const updated = await api.updateKanbanCard(selectedCard.id, { column });
              replaceCard(updated);
              await refreshDetail(updated.id);
            })
          }
          onDelete={() =>
            selectedCard &&
            runAction(async () => {
              await api.deleteKanbanCard(selectedCard.id);
              setCards((current) => current.filter((card) => card.id !== selectedCard.id));
              setSelectedId(null);
            })
          }
        />
      </div>
    </div>
  );
}
