import { useCallback, useEffect, useMemo, useState } from "react";
import { Brain, ChevronDown, ChevronRight, Pencil, RefreshCw, Save, Search, Trash2, User, X } from "lucide-react";
import { api } from "@/lib/api";
import type { MemoryEntry, MemoryResponse, MemoryStoreResponse } from "@/lib/api";
import { Toast } from "@/components/Toast";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/useToast";
import { useI18n } from "@/i18n";

const MEMORY_TEXT = {
  title: "Memory",
  userProfile: "User profile",
  notes: "Memory notes",
  builtInOnly: "built-in only",
  provider: "Provider",
  builtIn: "Built-in memory",
  directory: "Directory",
  refresh: "Refresh",
  snapshotNote: "Saved immediately. Changes apply to future sessions; current sessions keep their existing snapshot.",
  addEntry: "Add entry",
  emptyStore: "No entries yet.",
  entryCount: "entries",
  chars: "chars",
  edit: "Edit",
  delete: "Delete",
  saveFailed: "Failed to save",
  addFailed: "Failed to add",
  deleteFailed: "Failed to delete",
  target: "Target",
};

function previewForEntry(content: string) {
  const firstLine = content
    .split("\n")
    .map((line) => line.trim())
    .find(Boolean);
  return firstLine || content.trim() || "…";
}

function StoreSection({
  target,
  title,
  icon: Icon,
  store,
  expandedId,
  editingId,
  drafts,
  savingKey,
  searchValue,
  onSearchChange,
  onToggle,
  onStartEdit,
  onDraftChange,
  onSaveEdit,
  onDelete,
  m,
  t,
}: {
  target: "memory" | "user";
  title: string;
  icon: typeof Brain;
  store: MemoryStoreResponse;
  expandedId: string | null;
  editingId: string | null;
  drafts: Record<string, string>;
  savingKey: string | null;
  searchValue: string;
  onSearchChange: (target: "memory" | "user", value: string) => void;
  onToggle: (id: string) => void;
  onStartEdit: (entry: MemoryEntry) => void;
  onDraftChange: (id: string, value: string) => void;
  onSaveEdit: (id: string) => void;
  onDelete: (id: string) => void;
  m: typeof MEMORY_TEXT;
  t: ReturnType<typeof useI18n>["t"];
}) {
  const normalizedSearch = searchValue.trim().toLowerCase();
  const filteredEntries = normalizedSearch
    ? store.entries.filter((entry) => entry.content.toLowerCase().includes(normalizedSearch))
    : store.entries;

  return (
    <Card data-testid={`memory-store-${target}`}>
      <CardHeader className="py-3 px-4">
        <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:justify-between">
          <div className="flex items-center gap-2 min-w-0 flex-wrap">
            <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
            <CardTitle className="text-sm">{title}</CardTitle>
            <Badge tone="secondary" className="text-[10px]">
              {store.entry_count} {m.entryCount}
            </Badge>
            <Badge tone="outline" className="text-[10px]">
              {store.char_count}/{store.char_limit} {m.chars}
            </Badge>
          </div>
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              aria-label={`${title} ${t.common.search}`}
              placeholder={t.common.search}
              value={searchValue}
              onChange={(e) => onSearchChange(target, e.target.value)}
              className="pl-8 pr-7 h-8 text-xs"
            />
            {searchValue && (
              <button
                type="button"
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground cursor-pointer"
                onClick={() => onSearchChange(target, "")}
                aria-label={t.common.clear}
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid gap-2 px-4 pb-4">
        {filteredEntries.length === 0 ? (
          <div className="border border-border p-4 text-sm text-muted-foreground text-center">
            {normalizedSearch ? t.common.noResults : m.emptyStore}
          </div>
        ) : (
          filteredEntries.map((entry) => {
            const expanded = expandedId === entry.id;
            const editing = editingId === entry.id;
            const draftValue = drafts[entry.id] ?? entry.content;
            return (
              <div key={entry.id} className="border overflow-hidden border-border">
                <button
                  type="button"
                  className="flex w-full items-center justify-between p-3 cursor-pointer hover:bg-secondary/30 transition-colors"
                  onClick={() => onToggle(entry.id)}
                >
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    {expanded ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />}
                    <span className="font-medium text-sm truncate">{previewForEntry(entry.content)}</span>
                  </div>
                  <Badge tone="outline" className="text-[10px]">#{entry.index + 1}</Badge>
                </button>
                {expanded && (
                  <div className="border-t border-border bg-background/50 p-4 grid gap-3">
                    {editing ? (
                      <textarea
                        className="flex min-h-[140px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring font-courier leading-relaxed"
                        value={draftValue}
                        onChange={(e) => onDraftChange(entry.id, e.target.value)}
                      />
                    ) : (
                      <div className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">{entry.content}</div>
                    )}
                    <div className="flex items-center gap-2 flex-wrap">
                      {editing ? (
                        <Button
                          size="sm"
                          onClick={() => onSaveEdit(entry.id)}
                          disabled={savingKey === entry.id || !draftValue.trim()}
                          className="gap-1.5"
                        >
                          <Save className="h-3.5 w-3.5" />
                          {savingKey === entry.id ? t.common.saving : t.common.save}
                        </Button>
                      ) : (
                        <Button size="sm" ghost onClick={() => onStartEdit(entry)} className="gap-1.5">
                          <Pencil className="h-3.5 w-3.5" />
                          {m.edit}
                        </Button>
                      )}
                      <Button
                        size="sm"
                        ghost
                        onClick={() => onDelete(entry.id)}
                        disabled={savingKey === entry.id}
                        className="gap-1.5 text-destructive hover:text-destructive hover:bg-destructive/10"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        {m.delete}
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}

      </CardContent>
    </Card>
  );
}

export default function MemoryPage() {
  const [data, setData] = useState<MemoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingKey, setSavingKey] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [newEntry, setNewEntry] = useState("");
  const [newEntryTarget, setNewEntryTarget] = useState<"user" | "memory">("user");
  const [search, setSearch] = useState<{ memory: string; user: string }>({ memory: "", user: "" });
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const m = MEMORY_TEXT;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await api.getMemory());
    } catch (error) {
      showToast(String(error), "error");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    void load();
  }, [load]);

  const totalEntries = useMemo(() => {
    if (!data) return 0;
    return data.stores.user.entry_count + data.stores.memory.entry_count;
  }, [data]);

  const handleSaveEdit = async (entryId: string) => {
    const [target] = entryId.split(":") as ["memory" | "user", string];
    setSavingKey(entryId);
    try {
      const next = await api.updateMemoryEntry(target, entryId, drafts[entryId] ?? "");
      setData(next);
      setEditingId(null);
    } catch (error) {
      showToast(`${m.saveFailed}: ${error}`, "error");
    } finally {
      setSavingKey(null);
    }
  };

  const handleDelete = async (entryId: string) => {
    const [target] = entryId.split(":") as ["memory" | "user", string];
    setSavingKey(entryId);
    try {
      const next = await api.removeMemoryEntry(target, entryId);
      setData(next);
      setExpandedId((current) => (current === entryId ? null : current));
      setEditingId((current) => (current === entryId ? null : current));
    } catch (error) {
      showToast(`${m.deleteFailed}: ${error}`, "error");
    } finally {
      setSavingKey(null);
    }
  };

  const handleAdd = async () => {
    setSavingKey("new-entry");
    try {
      const next = await api.addMemoryEntry(newEntryTarget, newEntry);
      setData(next);
      setNewEntry("");
    } catch (error) {
      showToast(`${m.addFailed}: ${error}`, "error");
    } finally {
      setSavingKey(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex flex-col gap-4">
        <Toast toast={toast} />
        <div className="border border-destructive/30 bg-destructive/[0.06] p-4 text-sm text-destructive">
          {t.common.retry}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Toast toast={toast} />

      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <Brain className="h-5 w-5 text-muted-foreground shrink-0" />
          <h1 className="text-base font-semibold">{m.title}</h1>
          <Badge tone="secondary" className="text-xs">{totalEntries} {m.entryCount}</Badge>
        </div>
        <Button size="sm" ghost className="gap-1.5" onClick={() => void load()}>
          <RefreshCw className="h-3.5 w-3.5" />
          {m.refresh}
        </Button>
      </div>

      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="text-sm">{m.addEntry}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 px-4 pb-4">
          <div className="grid gap-2">
            <Label htmlFor="memory-new-entry">{m.addEntry}</Label>
            <textarea
              id="memory-new-entry"
              aria-label={`${m.title} ${m.addEntry}`}
              className="flex min-h-[120px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring font-courier leading-relaxed"
              value={newEntry}
              onChange={(e) => setNewEntry(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-[minmax(0,220px)_auto] gap-4 sm:items-end">
            <div className="grid gap-2">
              <Label htmlFor="memory-target-select">{m.target}</Label>
              <select
                id="memory-target-select"
                aria-label={`${m.title} ${m.target}`}
                className="h-9 rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                value={newEntryTarget}
                onChange={(event) => setNewEntryTarget(event.target.value as "user" | "memory")}
              >
                <option value="user">{m.userProfile}</option>
                <option value="memory">{m.notes}</option>
              </select>
            </div>
            <div>
              <Button size="sm" onClick={() => void handleAdd()} disabled={savingKey === "new-entry" || !newEntry.trim()} className="gap-1.5 w-full sm:w-auto">
                <Save className="h-3.5 w-3.5" />
                {savingKey === "new-entry" ? t.common.saving : m.addEntry}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="border border-warning/30 bg-warning/10 p-4 text-sm text-warning">
        {data.note || m.snapshotNote}
      </div>

      <StoreSection
        target="user"
        title={m.userProfile}
        icon={User}
        store={data.stores.user}
        expandedId={expandedId}
        editingId={editingId}
        drafts={drafts}
        savingKey={savingKey}
        searchValue={search.user}
        onSearchChange={(target, value) => setSearch((current) => ({ ...current, [target]: value }))}
        onToggle={(id) => setExpandedId((current) => (current === id ? null : id))}
        onStartEdit={(entry) => {
          setExpandedId(entry.id);
          setEditingId(entry.id);
          setDrafts((current) => ({ ...current, [entry.id]: entry.content }));
        }}
        onDraftChange={(id, value) => setDrafts((current) => ({ ...current, [id]: value }))}
        onSaveEdit={(id) => void handleSaveEdit(id)}
        onDelete={(id) => void handleDelete(id)}
        m={m}
        t={t}
      />

      <StoreSection
        target="memory"
        title={m.notes}
        icon={Brain}
        store={data.stores.memory}
        expandedId={expandedId}
        editingId={editingId}
        drafts={drafts}
        savingKey={savingKey}
        searchValue={search.memory}
        onSearchChange={(target, value) => setSearch((current) => ({ ...current, [target]: value }))}
        onToggle={(id) => setExpandedId((current) => (current === id ? null : id))}
        onStartEdit={(entry) => {
          setExpandedId(entry.id);
          setEditingId(entry.id);
          setDrafts((current) => ({ ...current, [entry.id]: entry.content }));
        }}
        onDraftChange={(id, value) => setDrafts((current) => ({ ...current, [id]: value }))}
        onSaveEdit={(id) => void handleSaveEdit(id)}
        onDelete={(id) => void handleDelete(id)}
        m={m}
        t={t}
      />
    </div>
  );
}
