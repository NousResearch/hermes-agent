import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent, PointerEvent } from "react";
import {
  ArrowUp,
  Braces,
  ChevronRight,
  Clock3,
  Columns2,
  Copy,
  Edit3,
  Eye,
  FileText,
  Folder,
  FolderOpen,
  GitBranch,
  Link as LinkIcon,
  ListTree,
  Network,
  Pin,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  ShieldCheck,
  Star,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  KnowledgeBacklinkItem,
  KnowledgeFileResponse,
  KnowledgeGraphResponse,
  KnowledgeSearchItem,
  KnowledgeStatusResponse,
  KnowledgeTreeItem,
} from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { PluginSlot } from "@/plugins";
import { cn } from "@/lib/utils";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Input } from "@/components/ui/input";

const ROOT_PATH = "";
const RECENT_STORAGE_KEY = "hermes.knowledge.recent";
const PINNED_STORAGE_KEY = "hermes.knowledge.pinned";
const MAX_RECENT_NOTES = 8;
const GRAPH_DEPTH = 2;
const GRAPH_LIMIT = 24;
const GLOBAL_GRAPH_LIMIT = 220;
const GLOBAL_GRAPH_EDGE_LIMIT = 900;

type StoredNote = {
  path: string;
  title: string;
};

type GraphMode = "global" | "local";

function readStoredNotes(key: string): StoredNote[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((item): item is StoredNote => (
        typeof item?.path === "string" && typeof item?.title === "string"
      ))
      .slice(0, MAX_RECENT_NOTES);
  } catch {
    return [];
  }
}

function writeStoredNotes(key: string, notes: StoredNote[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, JSON.stringify(notes.slice(0, MAX_RECENT_NOTES)));
}

function rememberNote(notes: StoredNote[], note: StoredNote): StoredNote[] {
  return [note, ...notes.filter((item) => item.path !== note.path)].slice(0, MAX_RECENT_NOTES);
}

function removeNote(notes: StoredNote[], path: string): StoredNote[] {
  return notes.filter((item) => item.path !== path);
}

function noteFromFile(file: KnowledgeFileResponse): StoredNote {
  return { path: file.path, title: file.title };
}

function breadcrumbParts(path: string): { label: string; path: string; type: "directory" | "file" }[] {
  const parts = path.split("/").filter(Boolean);
  return parts.map((part, index) => ({
    label: part,
    path: parts.slice(0, index + 1).join("/"),
    type: index === parts.length - 1 ? "file" : "directory",
  }));
}


function parentPath(path: string): string {
  const cleaned = path.replace(/\/+$/, "");
  if (!cleaned || !cleaned.includes("/")) return ROOT_PATH;
  return cleaned.split("/").slice(0, -1).join("/");
}

function candidateNotePath(link: string): string {
  if (/\.(md|canvas|base)$/i.test(link)) return link;
  return `${link}.md`;
}

function fileMeta(file: KnowledgeFileResponse | null): string {
  if (!file) return "No file";
  const kb = Math.max(1, Math.round(file.size / 1024));
  return `${kb} KB · ${file.links.length} links`;
}

function propertyValueLabel(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => propertyValueLabel(item)).join(", ");
  }
  if (value === null || value === undefined) return "-";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function scrollToHeading(slug: string): void {
  const target = document.getElementById(slug);
  target?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function KnowledgePage() {
  const [status, setStatus] = useState<KnowledgeStatusResponse | null>(null);
  const [treePath, setTreePath] = useState(ROOT_PATH);
  const [treeItems, setTreeItems] = useState<KnowledgeTreeItem[]>([]);
  const [selectedFile, setSelectedFile] = useState<KnowledgeFileResponse | null>(null);
  const [backlinks, setBacklinks] = useState<KnowledgeBacklinkItem[]>([]);
  const [graph, setGraph] = useState<KnowledgeGraphResponse | null>(null);
  const [graphMode, setGraphMode] = useState<GraphMode>("global");
  const [query, setQuery] = useState("");
  const [searchedQuery, setSearchedQuery] = useState("");
  const [results, setResults] = useState<KnowledgeSearchItem[]>([]);
  const [recentNotes, setRecentNotes] = useState<StoredNote[]>(() => readStoredNotes(RECENT_STORAGE_KEY));
  const [pinnedNotes, setPinnedNotes] = useState<StoredNote[]>(() => readStoredNotes(PINNED_STORAGE_KEY));
  const [openTabs, setOpenTabs] = useState<StoredNote[]>([]);
  const [graphZoom, setGraphZoom] = useState(1);
  const [viewMode, setViewMode] = useState<"preview" | "edit" | "split">("preview");
  const [editorContent, setEditorContent] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [fileLoading, setFileLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentLabel = treePath || "HermesAgent";
  const directoryItems = useMemo(
    () => treeItems.filter((item) => item.type === "directory"),
    [treeItems],
  );
  const fileItems = useMemo(
    () => treeItems.filter((item) => item.type === "file"),
    [treeItems],
  );
  const selectedIsPinned = Boolean(
    selectedFile && pinnedNotes.some((note) => note.path === selectedFile.path),
  );
  const selectedBreadcrumb = useMemo(
    () => breadcrumbParts(selectedFile?.path ?? ""),
    [selectedFile?.path],
  );
  const writeEnabled = Boolean(status?.write_enabled || selectedFile?.write_enabled);
  const editorDirty = Boolean(selectedFile && editorContent !== selectedFile.content);
  const frontmatterEntries = useMemo(
    () => Object.entries(selectedFile?.frontmatter ?? {}),
    [selectedFile?.frontmatter],
  );
  const writeStatusLabel = writeEnabled ? "WRITE ENABLED" : "READ ONLY";
  const searchEmpty = Boolean(searchedQuery && !searching && results.length === 0);

  const loadTree = useCallback((path: string) => {
    setLoading(true);
    setError(null);
    return api
      .getKnowledgeTree(path)
      .then((tree) => {
        setTreePath(tree.path);
        setTreeItems(tree.items);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  const loadStatus = useCallback(() => {
    return api
      .getKnowledgeStatus()
      .then(setStatus)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const loadGraph = useCallback((path: string, mode: GraphMode) => {
    const request = mode === "global"
      ? api.getKnowledgeGlobalGraph({ limit: GLOBAL_GRAPH_LIMIT, edgeLimit: GLOBAL_GRAPH_EDGE_LIMIT })
      : api.getKnowledgeGraph(path, { depth: GRAPH_DEPTH, limit: GRAPH_LIMIT });
    return request.then(setGraph);
  }, []);

  const changeGraphMode = useCallback((mode: GraphMode) => {
    setGraphMode(mode);
    if (selectedFile) {
      void loadGraph(selectedFile.path, mode).catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });
    }
  }, [loadGraph, selectedFile]);

  const openFile = useCallback((path: string, options: { syncUrl?: boolean } = {}) => {
    setFileLoading(true);
    setError(null);
    api
      .getKnowledgeFile(path)
      .then((file) => {
        setSelectedFile(file);
        setEditorContent(file.content);
        setSaveMessage(null);
        setOpenTabs((current) => rememberNote(current, noteFromFile(file)).slice(0, 6));
        setRecentNotes((current) => {
          const next = rememberNote(current, noteFromFile(file));
          writeStoredNotes(RECENT_STORAGE_KEY, next);
          return next;
        });
        if (options.syncUrl !== false) {
          const url = new URL(window.location.href);
          url.searchParams.set("note", file.path);
          window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
        }
        void loadTree(parentPath(file.path));
        return Promise.all([
          api.getKnowledgeBacklinks(file.path).then((resp) => setBacklinks(resp.items)),
          loadGraph(file.path, graphMode),
        ]);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setFileLoading(false));
  }, [graphMode, loadGraph, loadTree]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void loadStatus();
      const initialNote = new URLSearchParams(window.location.search).get("note");
      if (initialNote) {
        openFile(initialNote, { syncUrl: false });
      } else {
        void loadTree(ROOT_PATH);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [loadStatus, loadTree, openFile]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        const searchInput = document.getElementById("knowledge-search-input") as HTMLInputElement | null;
        searchInput?.focus();
        searchInput?.select();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const onSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const term = query.trim();
    setSearchedQuery(term);
    if (!term) {
      setResults([]);
      return;
    }
    setSearching(true);
    setError(null);
    api
      .searchKnowledge(term)
      .then((resp) => setResults(resp.items))
      .catch((err) => {
        setResults([]);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => setSearching(false));
  };

  const openLinkedNote = (link: string) => {
    api
      .resolveKnowledgeLink(link, selectedFile?.path ?? "")
      .then((resolved) => openFile(resolved.path))
      .catch(() => openFile(candidateNotePath(link)));
  };

  const togglePinnedNote = () => {
    if (!selectedFile) return;
    const note = noteFromFile(selectedFile);
    setPinnedNotes((current) => {
      const next = current.some((item) => item.path === note.path)
        ? removeNote(current, note.path)
        : rememberNote(current, note);
      writeStoredNotes(PINNED_STORAGE_KEY, next);
      return next;
    });
  };

  const clearRecentNotes = () => {
    setRecentNotes([]);
    writeStoredNotes(RECENT_STORAGE_KEY, []);
  };

  const clearSearch = () => {
    setQuery("");
    setSearchedQuery("");
    setResults([]);
  };

  const copySelectedPath = useCallback(() => {
    if (!selectedFile) return;
    const text = selectedFile.path;
    if (navigator.clipboard?.writeText) {
      void navigator.clipboard
        .writeText(text)
        .then(() => setSaveMessage("Path copied"))
        .catch(() => setSaveMessage(text));
      return;
    }
    setSaveMessage(text);
  }, [selectedFile]);

  const closeTab = (path: string) => {
    setOpenTabs((current) => current.filter((note) => note.path !== path));
  };

  const saveSelectedFile = useCallback(() => {
    if (!selectedFile || !writeEnabled || !editorDirty) return;
    setSaving(true);
    setError(null);
    setSaveMessage(null);
    api
      .saveKnowledgeFile(selectedFile.path, editorContent, selectedFile.modified)
      .then((file) => {
        setSelectedFile(file);
        setEditorContent(file.content);
        setSaveMessage(file.backup_path ? `Saved · backup ${file.backup_path}` : "Saved");
        return Promise.all([
          api.getKnowledgeBacklinks(file.path).then((resp) => setBacklinks(resp.items)),
          loadGraph(file.path, graphMode),
        ]);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setSaving(false));
  }, [editorContent, editorDirty, graphMode, loadGraph, selectedFile, writeEnabled]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveSelectedFile();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [saveSelectedFile]);

  useEffect(() => {
    if (!editorDirty) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [editorDirty]);

  return (
    <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col gap-3">
      <PluginSlot name="knowledge:top" />

      <div className="flex min-w-0 flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Badge tone={writeEnabled ? "warning" : "success"} className="gap-1 text-[10px]">
            <ShieldCheck className="h-3 w-3" />
            {writeStatusLabel}
          </Badge>
          {status ? (
            <Badge tone="secondary" className="max-w-full truncate text-[10px]">
              {status.vault_name} · {status.safe_file_count} files
            </Badge>
          ) : null}
          {selectedFile ? (
            <Badge tone="secondary" className="max-w-full truncate text-[10px]">
              {fileMeta(selectedFile)}
            </Badge>
          ) : null}
        </div>

        <form onSubmit={onSearch} className="flex min-w-0 gap-2 lg:w-[28rem]">
          <Input
            id="knowledge-search-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search vault"
            className="min-w-0"
          />
          <Button type="submit" size="sm" className="shrink-0 gap-2" disabled={searching}>
            {searching ? <Spinner /> : <Search className="h-3.5 w-3.5" />}
            Search
          </Button>
          {(searchedQuery || results.length > 0) ? (
            <Button type="button" ghost size="icon" onClick={clearSearch} aria-label="Clear search">
              <X className="h-3.5 w-3.5" />
            </Button>
          ) : null}
        </form>
      </div>

      {error ? (
        <div className="border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm normal-case text-destructive">
          {error}
        </div>
      ) : null}

      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[18rem_minmax(0,1fr)_20rem]">
        <aside className="flex min-h-[16rem] min-w-0 flex-col border border-current/20 bg-background-base/45 lg:min-h-0">
          <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-current/15 px-3">
            <div className="min-w-0 truncate text-xs font-bold tracking-[0.12em] text-midground">
              {currentLabel}
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <Button ghost size="icon" onClick={() => void loadTree(parentPath(treePath))} disabled={!treePath}>
                <ArrowUp className="h-3.5 w-3.5" />
              </Button>
              <Button ghost size="icon" onClick={() => void loadTree(treePath)} disabled={loading}>
                {loading ? <Spinner /> : <RefreshCw className="h-3.5 w-3.5" />}
              </Button>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-auto p-2">
            {[...directoryItems, ...fileItems].map((item) => (
              <button
                key={item.path}
                type="button"
                onClick={() => item.type === "directory" ? void loadTree(item.path) : openFile(item.path)}
                className={cn(
                  "flex w-full min-w-0 items-center gap-2 px-2 py-1.5 text-left text-xs tracking-[0.04em]",
                  "text-midground/70 hover:bg-midground/10 hover:text-midground",
                  selectedFile?.path === item.path && "bg-midground/10 text-midground",
                )}
              >
                {item.type === "directory" ? <Folder className="h-3.5 w-3.5 shrink-0" /> : <FileText className="h-3.5 w-3.5 shrink-0" />}
                <span className="min-w-0 truncate normal-case">{item.name}</span>
              </button>
            ))}
            {!loading && directoryItems.length === 0 && fileItems.length === 0 ? (
              <EmptyLine label="No notes in this folder" />
            ) : null}
          </div>
        </aside>

        <main className="flex min-h-[24rem] min-w-0 flex-col border border-current/20 bg-background-base/35 lg:min-h-0">
          <div className="flex h-11 shrink-0 items-center gap-2 border-b border-current/15 px-3">
            <FileText className="h-4 w-4 shrink-0 text-midground/60" />
            <div className="min-w-0 truncate text-sm font-bold tracking-[0.08em] text-midground">
              {selectedFile?.title ?? "Select a note"}
            </div>
            {selectedFile ? (
              <div className="ml-auto flex shrink-0 items-center gap-1">
                <Button ghost size="icon" onClick={() => void loadTree(parentPath(selectedFile.path))} aria-label="Open parent folder">
                  <FolderOpen className="h-3.5 w-3.5" />
                </Button>
                <Button ghost size="icon" onClick={() => openFile(selectedFile.path)} aria-label="Reload note">
                  <RefreshCw className="h-3.5 w-3.5" />
                </Button>
                <Button ghost size="icon" onClick={copySelectedPath} aria-label="Copy note path">
                  <Copy className="h-3.5 w-3.5" />
                </Button>
                <ModeButton active={viewMode === "preview"} label="Preview" onClick={() => setViewMode("preview")} icon={Eye} />
                <ModeButton active={viewMode === "edit"} label="Edit" onClick={() => setViewMode("edit")} icon={Edit3} />
                <ModeButton active={viewMode === "split"} label="Split" onClick={() => setViewMode("split")} icon={Columns2} />
                <Button ghost size="icon" onClick={togglePinnedNote} aria-label={selectedIsPinned ? "Unpin note" : "Pin note"}>
                  {selectedIsPinned ? <Star className="h-3.5 w-3.5 fill-current" /> : <Pin className="h-3.5 w-3.5" />}
                </Button>
                <Button
                  size="sm"
                  className="gap-2"
                  onClick={saveSelectedFile}
                  disabled={!writeEnabled || !editorDirty || saving}
                >
                  {saving ? <Spinner /> : <Save className="h-3.5 w-3.5" />}
                  Save
                </Button>
              </div>
            ) : null}
            {fileLoading ? <Spinner className={cn("shrink-0", !selectedFile && "ml-auto")} /> : null}
          </div>

          {openTabs.length > 0 ? (
            <div className="flex h-10 shrink-0 items-center gap-1 overflow-x-auto border-b border-current/10 px-2 normal-case">
              {openTabs.map((tab) => (
                <div
                  key={tab.path}
                  className={cn(
                    "flex max-w-56 shrink-0 items-center border border-current/10 bg-background-base/30 text-xs",
                    selectedFile?.path === tab.path && "bg-midground/10 text-midground",
                  )}
                >
                  <button type="button" onClick={() => openFile(tab.path)} className="min-w-0 truncate px-2 py-1.5">
                    {tab.title}
                  </button>
                  <button type="button" onClick={() => closeTab(tab.path)} className="px-1.5 text-midground/45 hover:text-midground" aria-label={`Close ${tab.title}`}>
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          ) : null}

          {selectedFile && selectedBreadcrumb.length > 0 ? (
            <div className="flex h-9 shrink-0 items-center gap-1 overflow-x-auto border-b border-current/10 px-3 text-[10px] text-midground/55 normal-case">
              <button type="button" onClick={() => void loadTree(ROOT_PATH)} className="shrink-0 hover:text-midground">
                HermesAgent
              </button>
              {selectedBreadcrumb.map((part) => (
                <span key={part.path} className="flex shrink-0 items-center gap-1">
                  <ChevronRight className="h-3 w-3 opacity-45" />
                  <button
                    type="button"
                    onClick={() => part.type === "directory" ? void loadTree(part.path) : openFile(part.path)}
                    className="max-w-48 truncate hover:text-midground"
                  >
                    {part.label}
                  </button>
                </span>
              ))}
            </div>
          ) : null}

          <div className="min-h-0 flex-1 overflow-auto p-4 normal-case">
            {selectedFile ? (
              <div className="mx-auto flex max-w-4xl flex-col gap-4">
                <div className="flex flex-wrap gap-2">
                  <Badge tone="secondary" className="max-w-full truncate text-[10px]">
                    {selectedFile.path}
                  </Badge>
                  <Badge tone={writeEnabled ? "success" : "secondary"} className="text-[10px]">
                    {writeEnabled ? "WRITE ENABLED" : "READ ONLY"}
                  </Badge>
                  {editorDirty ? <Badge tone="warning" className="text-[10px]">UNSAVED</Badge> : null}
                  {selectedFile.truncated ? <Badge tone="warning">TRUNCATED</Badge> : null}
                  {saveMessage ? <Badge tone="success" className="max-w-full truncate text-[10px]">{saveMessage}</Badge> : null}
                </div>
                <NoteWorkspace
                  mode={viewMode}
                  content={editorContent}
                  readOnly={!writeEnabled}
                  onChange={setEditorContent}
                  onWikiLink={openLinkedNote}
                />
              </div>
            ) : (
              <div className="flex h-full min-h-[16rem] items-center justify-center text-sm text-midground/40">
                No note selected
              </div>
            )}
          </div>
        </main>

        <aside className="flex min-h-[18rem] min-w-0 flex-col gap-3 overflow-auto lg:min-h-0">
          {pinnedNotes.length > 0 ? (
            <section className="border border-current/20 bg-background-base/35">
              <PanelHeader icon={Star} title="Pinned" meta={`${pinnedNotes.length}`} />
              <div className="max-h-44 overflow-auto p-2">
                {pinnedNotes.map((note) => (
                  <SideButton key={note.path} title={note.title} subtitle={note.path} onClick={() => openFile(note.path)} />
                ))}
              </div>
            </section>
          ) : null}

          {recentNotes.length > 0 ? (
            <section className="border border-current/20 bg-background-base/35">
              <PanelHeader icon={Clock3} title="Recent" meta={`${recentNotes.length}`} action={clearRecentNotes} />
              <div className="max-h-44 overflow-auto p-2">
                {recentNotes.map((note) => (
                  <SideButton key={note.path} title={note.title} subtitle={note.path} onClick={() => openFile(note.path)} />
                ))}
              </div>
            </section>
          ) : null}

          {results.length > 0 || searchedQuery ? (
            <section className="border border-current/20 bg-background-base/35">
              <PanelHeader icon={Search} title="Search results" meta={`${results.length}`} action={clearSearch} />
              <div className="max-h-64 overflow-auto p-2">
                {results.map((item) => (
                  <SideButton
                    key={item.path}
                    title={item.title}
                    subtitle={item.path}
                    detail={item.snippet}
                    onClick={() => openFile(item.path)}
                  />
                ))}
                {searchEmpty ? <EmptyLine label={`No results for ${searchedQuery}`} /> : null}
              </div>
            </section>
          ) : null}

          <section className="border border-current/20 bg-background-base/35">
            <PanelHeader icon={ListTree} title="Outline" meta={`${selectedFile?.headings.length ?? 0}`} />
            <div className="max-h-56 overflow-auto p-2">
              {(selectedFile?.headings ?? []).map((heading) => (
                <button
                  key={`${heading.slug}-${heading.line}`}
                  type="button"
                  onClick={() => scrollToHeading(heading.slug)}
                  className="flex w-full min-w-0 items-center gap-2 px-2 py-1.5 text-left hover:bg-midground/10"
                  style={{ paddingLeft: `${8 + Math.max(0, heading.level - 1) * 10}px` }}
                >
                  <span className="w-7 shrink-0 text-[10px] text-midground/35 normal-case">L{heading.line}</span>
                  <span className="min-w-0 flex-1 truncate text-xs text-midground normal-case">{heading.title}</span>
                </button>
              ))}
              {selectedFile && selectedFile.headings.length === 0 ? <EmptyLine label="No headings" /> : null}
            </div>
          </section>

          <section className="border border-current/20 bg-background-base/35">
            <PanelHeader icon={Braces} title="Properties" meta={`${frontmatterEntries.length}`} />
            <div className="max-h-56 overflow-auto p-2">
              {frontmatterEntries.map(([key, value]) => (
                <div key={key} className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-2 px-2 py-1.5 text-xs normal-case">
                  <span className="truncate text-midground/45">{key}</span>
                  <span className="min-w-0 truncate text-midground">{propertyValueLabel(value)}</span>
                </div>
              ))}
              {selectedFile && frontmatterEntries.length === 0 ? <EmptyLine label="No properties" /> : null}
            </div>
          </section>

          <section className="border border-current/20 bg-background-base/35">
            <PanelHeader icon={LinkIcon} title="Links" meta={`${selectedFile?.links.length ?? 0}`} />
            <div className="max-h-56 overflow-auto p-2">
              {(selectedFile?.links ?? []).map((link) => (
                <SideButton key={link} title={link} subtitle="outgoing" onClick={() => openLinkedNote(link)} />
              ))}
              {selectedFile && selectedFile.links.length === 0 ? <EmptyLine label="No outgoing links" /> : null}
            </div>
          </section>

          <section className="border border-current/20 bg-background-base/35">
            <PanelHeader icon={GitBranch} title="Backlinks" meta={`${backlinks.length}`} />
            <div className="max-h-64 overflow-auto p-2">
              {backlinks.map((item) => (
                <SideButton
                  key={item.path}
                  title={item.title}
                  subtitle={item.path}
                  detail={item.snippet}
                  onClick={() => openFile(item.path)}
                />
              ))}
              {selectedFile && backlinks.length === 0 ? <EmptyLine label="No backlinks" /> : null}
            </div>
          </section>

          <section className="border border-current/20 bg-background-base/35">
            <PanelHeader
              icon={Network}
              title="Graph"
              meta={graphMode === "global"
                ? `GLOBAL · ${graph?.nodes.length ?? 0}/${graph?.node_count ?? 0} · ${graph?.edges.length ?? 0}`
                : `LOCAL D${graph?.depth ?? GRAPH_DEPTH} · ${graph?.nodes.length ?? 0} · ${graph?.edges.length ?? 0}`}
            />
            <div className="p-2">
              <div className="mb-2 grid grid-cols-2 gap-1">
                <Button
                  ghost={graphMode !== "global"}
                  size="sm"
                  onClick={() => changeGraphMode("global")}
                >
                  Global
                </Button>
                <Button
                  ghost={graphMode !== "local"}
                  size="sm"
                  onClick={() => changeGraphMode("local")}
                  disabled={!selectedFile}
                >
                  Local
                </Button>
              </div>
              {graph && graph.nodes.length > 1 ? (
                <GraphMap
                  key={`${graphMode}:${graph.path}:${graph.nodes.length}`}
                  graph={graph}
                  selectedPath={selectedFile?.path ?? ""}
                  zoom={graphZoom}
                  onZoomIn={() => setGraphZoom((value) => Math.min(1.8, value + 0.2))}
                  onZoomOut={() => setGraphZoom((value) => Math.max(0.7, value - 0.2))}
                  onOpen={openFile}
                />
              ) : selectedFile ? (
                <EmptyLine label={graphMode === "global" ? "No global graph" : "No local graph"} />
              ) : null}
            </div>
          </section>
        </aside>
      </div>

      <PluginSlot name="knowledge:bottom" />
    </div>
  );
}

function ModeButton({
  active,
  label,
  icon: Icon,
  onClick,
}: {
  active: boolean;
  label: string;
  icon: typeof Search;
  onClick: () => void;
}) {
  return (
    <Button ghost size="icon" onClick={onClick} aria-label={label} className={active ? "bg-midground/10" : undefined}>
      <Icon className="h-3.5 w-3.5" />
    </Button>
  );
}

function NoteWorkspace({
  mode,
  content,
  readOnly,
  onChange,
  onWikiLink,
}: {
  mode: "preview" | "edit" | "split";
  content: string;
  readOnly: boolean;
  onChange: (value: string) => void;
  onWikiLink: (target: string) => void;
}) {
  if (mode === "preview") {
    return <Markdown content={content} onWikiLink={onWikiLink} />;
  }

  const editor = (
    <textarea
      value={content}
      onChange={(event) => onChange(event.target.value)}
      readOnly={readOnly}
      spellCheck={false}
      className="min-h-[28rem] w-full resize-none border border-current/15 bg-background-base/60 p-3 font-mono text-sm leading-relaxed text-midground outline-none focus:border-current/35"
    />
  );

  if (mode === "edit") {
    return editor;
  }

  return (
    <div className="grid min-h-[28rem] gap-3 xl:grid-cols-2">
      {editor}
      <div className="min-h-[28rem] overflow-auto border border-current/15 bg-background-base/35 p-3">
        <Markdown content={content} onWikiLink={onWikiLink} />
      </div>
    </div>
  );
}

function PanelHeader({
  icon: Icon,
  title,
  meta,
  action,
}: {
  icon: typeof Search;
  title: string;
  meta: string;
  action?: () => void;
}) {
  return (
    <div className="flex h-10 items-center gap-2 border-b border-current/15 px-3">
      <Icon className="h-3.5 w-3.5 shrink-0 text-midground/55" />
      <span className="min-w-0 flex-1 truncate text-xs font-bold tracking-[0.1em] text-midground">{title}</span>
      <Badge tone="secondary" className="text-[10px]">{meta}</Badge>
      {action ? (
        <button type="button" onClick={action} className="text-midground/45 hover:text-midground" aria-label={`Clear ${title}`}>
          <X className="h-3 w-3" />
        </button>
      ) : null}
    </div>
  );
}

function SideButton({
  title,
  subtitle,
  detail,
  onClick,
}: {
  title: string;
  subtitle: string;
  detail?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full min-w-0 flex-col gap-0.5 px-2 py-1.5 text-left hover:bg-midground/10"
    >
      <span className="min-w-0 max-w-full truncate text-xs text-midground normal-case">{title}</span>
      <span className="min-w-0 max-w-full truncate text-[10px] text-midground/45 normal-case">{subtitle}</span>
      {detail ? (
        <span className="line-clamp-2 text-[10px] leading-snug text-midground/55 normal-case">
          {detail}
        </span>
      ) : null}
    </button>
  );
}

type GraphPosition = { x: number; y: number };
type GraphDragState = { path: string; moved: boolean };

function GraphMap({
  graph,
  selectedPath,
  zoom,
  onZoomIn,
  onZoomOut,
  onOpen,
}: {
  graph: KnowledgeGraphResponse;
  selectedPath: string;
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onOpen: (path: string) => void;
}) {
  const [customPositions, setCustomPositions] = useState<Record<string, GraphPosition>>({});
  const [dragState, setDragState] = useState<GraphDragState | null>(null);

  const isGlobalGraph = graph.mode === "global";

  const baseNodes = useMemo(() => {
    const centerX = 150;
    const centerY = 150;
    const selected = graph.nodes.find((node) => node.path === selectedPath);
    const sortedNodes = [...graph.nodes].sort((a, b) => (b.degree ?? 0) - (a.degree ?? 0));
    const focusNode = selected ?? sortedNodes[0];
    const outer = graph.nodes.filter((node) => node.path !== focusNode?.path);
    const positioned = outer.map((node, index) => {
      if (isGlobalGraph) {
        const normalized = Math.sqrt((index + 1) / Math.max(1, outer.length));
        const angle = index * 2.399963229728653;
        const radius = 18 + normalized * 124;
        return {
          ...node,
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
        };
      }
      const angle = (Math.PI * 2 * index) / Math.max(1, outer.length) - Math.PI / 2;
      return {
        ...node,
        x: centerX + Math.cos(angle) * 82,
        y: centerY + Math.sin(angle) * 82,
      };
    });
    return focusNode
      ? [{ ...focusNode, x: centerX, y: centerY }, ...positioned]
      : positioned;
  }, [graph.nodes, isGlobalGraph, selectedPath]);

  const nodes = useMemo(
    () => baseNodes.map((node) => ({ ...node, ...(customPositions[node.path] ?? {}) })),
    [baseNodes, customPositions],
  );

  const nodeMap = useMemo(
    () => new Map(nodes.map((node) => [node.path, node])),
    [nodes],
  );

  const viewSize = 300 / zoom;
  const viewOffset = (300 - viewSize) / 2;

  const graphPointFromEvent = (event: PointerEvent<SVGSVGElement>): GraphPosition => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = viewOffset + ((event.clientX - rect.left) / rect.width) * viewSize;
    const y = viewOffset + ((event.clientY - rect.top) / rect.height) * viewSize;
    return {
      x: Math.max(18, Math.min(282, x)),
      y: Math.max(18, Math.min(282, y)),
    };
  };

  const onGraphPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!dragState) return;
    const next = graphPointFromEvent(event);
    setCustomPositions((current) => ({ ...current, [dragState.path]: next }));
    setDragState((current) => current ? { ...current, moved: true } : current);
  };

  const onNodePointerDown = (event: PointerEvent<SVGGElement>, path: string) => {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragState({ path, moved: false });
  };

  const onNodePointerUp = (event: PointerEvent<SVGGElement>, path: string) => {
    event.preventDefault();
    event.stopPropagation();
    if (dragState?.path === path && !dragState.moved) {
      onOpen(path);
    }
    setDragState(null);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-end gap-1">
        <Button ghost size="icon" onClick={() => setCustomPositions({})} aria-label="Reset graph layout">
          <RotateCcw className="h-3.5 w-3.5" />
        </Button>
        <Button ghost size="icon" onClick={onZoomOut} aria-label="Zoom graph out">
          <ZoomOut className="h-3.5 w-3.5" />
        </Button>
        <Badge tone="secondary" className="text-[10px]">{Math.round(zoom * 100)}%</Badge>
        <Button ghost size="icon" onClick={onZoomIn} aria-label="Zoom graph in">
          <ZoomIn className="h-3.5 w-3.5" />
        </Button>
      </div>
      <svg
        role="img"
        aria-label="Knowledge graph"
        viewBox={`${viewOffset} ${viewOffset} ${viewSize} ${viewSize}`}
        className={cn(
          "w-full touch-none select-none border border-current/10 bg-background-base/45",
          isGlobalGraph ? "h-72" : "h-56",
        )}
        onPointerMove={onGraphPointerMove}
        onPointerUp={() => setDragState(null)}
        onPointerLeave={() => setDragState(null)}
      >
        {graph.edges.map((edge) => {
          const source = nodeMap.get(edge.source);
          const target = nodeMap.get(edge.target);
          if (!source || !target) return null;
          return (
            <line
              key={`${edge.source}->${edge.target}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              className="stroke-midground/25"
              strokeWidth="1.2"
            />
          );
        })}
        {nodes.map((node, index) => {
          const active = node.path === selectedPath;
          const degree = node.degree ?? 1;
          const radius = isGlobalGraph
            ? Math.min(active ? 14 : 9, 2.8 + Math.sqrt(Math.max(1, degree)) * 1.8)
            : active ? 18 : 12;
          const showLabel = !isGlobalGraph || active || (index < 10 && degree > 1);
          return (
            <g
              key={node.id}
              className="cursor-grab active:cursor-grabbing"
              onPointerDown={(event) => onNodePointerDown(event, node.path)}
              onPointerUp={(event) => onNodePointerUp(event, node.path)}
            >
              <title>{`${node.label} · ${node.path}`}</title>
              <circle
                cx={node.x}
                cy={node.y}
                r={radius}
                className={active ? "fill-primary/80 stroke-primary" : "fill-midground/35 stroke-midground/55"}
                strokeWidth={active ? "2" : "1.2"}
              />
              {showLabel ? (
                <text
                  x={node.x}
                  y={node.y + radius + 10}
                  textAnchor="middle"
                  className="pointer-events-none fill-midground text-[8px]"
                >
                  {node.label.length > 18 ? `${node.label.slice(0, 17)}...` : node.label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div className="max-h-36 overflow-auto">
        {graph.nodes.map((node) => (
          <SideButton key={node.id} title={node.label} subtitle={node.path} onClick={() => onOpen(node.path)} />
        ))}
      </div>
    </div>
  );
}

function EmptyLine({ label }: { label: string }) {
  return <div className="px-2 py-2 text-xs normal-case text-midground/35">{label}</div>;
}
