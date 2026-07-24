import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent as ReactDragEvent,
} from "react";
import {
  ArrowLeft,
  ArrowUp,
  ChevronRight,
  Code2,
  Download,
  Eye,
  FileIcon,
  FileText,
  Folder,
  FolderOpen,
  FolderPlus,
  ImageIcon,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@nous-research/ui/ui/components/dialog";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { FilePreview } from "@/components/files/FilePreview";
import { FileTree } from "@/components/files/FileTree";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type {
  ManagedFileEntry,
  ManagedFileReadResponse,
  ManagedFilesResponse,
} from "@/lib/api";
import {
  buildBreadcrumbs,
  decodeTextDataUrl,
  pathName,
  previewKind,
} from "@/lib/file-browser";
import { PluginSlot } from "@/plugins";

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

type DirectoryEntries = Record<string, ManagedFileEntry[]>;
type PreviewMode = "rendered" | "source";

function joinPath(base: string, name: string): string {
  const cleanName = name.trim().replace(/^[\\/]+/, "");
  if (!cleanName) return base;
  const separator = base.includes("\\") && !base.includes("/") ? "\\" : "/";
  if (!base || base.endsWith("/") || base.endsWith("\\")) return `${base}${cleanName}`;
  return `${base}${separator}${cleanName}`;
}

function isPathWithin(root: string, path: string): boolean {
  const separator = root.includes("\\") && !root.includes("/") ? "\\" : "/";
  const cleanRoot = root.replace(/[\\/]+$/, "");
  return path === cleanRoot || path.startsWith(`${cleanRoot}${separator}`);
}

function formatBytes(size: number | null): string {
  if (size === null) return "-";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function downloadDataUrl(dataUrl: string, name: string) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = name || "download";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function transferHasFiles(event: ReactDragEvent<HTMLElement>): boolean {
  return Array.from(event.dataTransfer.types).includes("Files");
}

export default function FilesPage() {
  const { toast, showToast } = useToast();
  const { setAfterTitle, setEnd } = usePageHeader();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragDepthRef = useRef(0);
  const currentRequestRef = useRef(0);
  const previewRequestRef = useRef(0);
  const [currentPath, setCurrentPath] = useState<string | undefined>(undefined);
  const [pathInput, setPathInput] = useState("");
  const [listing, setListing] = useState<ManagedFilesResponse | null>(null);
  const [treeRoot, setTreeRoot] = useState<string | null>(null);
  const [entriesByDirectory, setEntriesByDirectory] = useState<DirectoryEntries>({});
  const [expandedDirectories, setExpandedDirectories] = useState<Set<string>>(new Set());
  const [loadingDirectories, setLoadingDirectories] = useState<Set<string>>(new Set());
  const [selectedEntry, setSelectedEntry] = useState<ManagedFileEntry | null>(null);
  const [selectedFile, setSelectedFile] = useState<ManagedFileReadResponse | null>(null);
  const [previewText, setPreviewText] = useState("");
  const [previewMode, setPreviewMode] = useState<PreviewMode>("rendered");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [draggingFiles, setDraggingFiles] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [folderName, setFolderName] = useState("");
  const [pendingDelete, setPendingDelete] = useState<ManagedFileEntry | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activePath = listing?.path ?? currentPath ?? "";
  const canChangePath = listing?.can_change_path ?? false;
  const canUpload = Boolean(activePath) && !uploading;
  const selectedKind = selectedEntry
    ? previewKind(selectedEntry.name, selectedEntry.mime_type, selectedEntry.size ?? 0)
    : null;
  const breadcrumbRoot = listing?.locked_root ?? treeRoot ?? activePath;
  const breadcrumbs = useMemo(
    () => (breadcrumbRoot && activePath ? buildBreadcrumbs(breadcrumbRoot, activePath) : []),
    [activePath, breadcrumbRoot],
  );

  const loadCurrentDirectory = useCallback(async (path?: string) => {
    const requestId = ++currentRequestRef.current;
    setLoading(true);
    setError(null);
    try {
      const result = await api.listFiles(path);
      if (requestId !== currentRequestRef.current) return;
      setListing(result);
      setCurrentPath(result.path);
      setPathInput(result.path);
      setEntriesByDirectory((current) => ({ ...current, [result.path]: result.entries }));
      setExpandedDirectories((current) => new Set(current).add(result.path));
      setTreeRoot((root) => {
        if (result.locked_root) return result.locked_root;
        return root && isPathWithin(root, result.path) ? root : result.path;
      });
    } catch (e) {
      if (requestId === currentRequestRef.current) setError(String(e));
    } finally {
      if (requestId === currentRequestRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadCurrentDirectory(currentPath);
  }, [currentPath, loadCurrentDirectory]);

  useEffect(() => {
    const headerPath = listing?.locked_root ?? listing?.path ?? currentPath ?? "Files";
    setAfterTitle(
      <Badge tone="outline" className="max-w-[22rem] truncate text-xs" title={headerPath}>
        {headerPath}
      </Badge>,
    );
    setEnd(
      <Button
        ghost
        size="icon"
        type="button"
        onClick={() => void loadCurrentDirectory(currentPath)}
        disabled={loading}
        aria-label="Refresh files"
      >
        {loading ? <Spinner /> : <RefreshCw />}
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [currentPath, listing, loadCurrentDirectory, loading, setAfterTitle, setEnd]);

  const ensureDirectoryLoaded = async (path: string) => {
    if (entriesByDirectory[path] || loadingDirectories.has(path)) return;
    setLoadingDirectories((current) => new Set(current).add(path));
    try {
      const result = await api.listFiles(path);
      setEntriesByDirectory((current) => ({ ...current, [result.path]: result.entries }));
    } catch (e) {
      showToast(`Could not open folder: ${e}`, "error");
    } finally {
      setLoadingDirectories((current) => {
        const next = new Set(current);
        next.delete(path);
        return next;
      });
    }
  };

  const clearSelection = () => {
    previewRequestRef.current += 1;
    setSelectedEntry(null);
    setSelectedFile(null);
    setPreviewText("");
    setPreviewError(null);
    setPreviewLoading(false);
  };

  const openDirectory = (path: string) => {
    clearSelection();
    setExpandedDirectories((current) => new Set(current).add(path));
    if (path === currentPath) void loadCurrentDirectory(path);
    else setCurrentPath(path);
  };

  const toggleDirectory = (path: string) => {
    if (expandedDirectories.has(path)) {
      setExpandedDirectories((current) => {
        const next = new Set(current);
        next.delete(path);
        return next;
      });
      return;
    }
    setExpandedDirectories((current) => new Set(current).add(path));
    void ensureDirectoryLoaded(path);
  };

  const openFile = async (entry: ManagedFileEntry) => {
    if (entry.is_directory) return;
    const kind = previewKind(entry.name, entry.mime_type, entry.size ?? 0);
    const requestId = ++previewRequestRef.current;
    setSelectedEntry(entry);
    setSelectedFile(null);
    setPreviewText("");
    setPreviewMode(kind === "markdown" ? "rendered" : "source");
    setPreviewError(null);

    if (kind === "large" || kind === "binary") {
      setPreviewLoading(false);
      return;
    }

    setPreviewLoading(true);
    try {
      const file = await api.readFile(entry.path);
      if (requestId !== previewRequestRef.current) return;
      setSelectedFile(file);
      if (kind === "markdown" || kind === "text") {
        setPreviewText(decodeTextDataUrl(file.data_url));
      }
    } catch (e) {
      if (requestId === previewRequestRef.current) setPreviewError(`Preview failed: ${e}`);
    } finally {
      if (requestId === previewRequestRef.current) setPreviewLoading(false);
    }
  };

  const goToPath = async () => {
    const nextPath = pathInput.trim();
    if (!nextPath) {
      showToast("Path required", "error");
      return;
    }
    clearSelection();
    setTreeRoot(null);
    setEntriesByDirectory({});
    setExpandedDirectories(new Set());
    if (nextPath === currentPath) await loadCurrentDirectory(nextPath);
    else setCurrentPath(nextPath);
  };

  const createDirectory = async () => {
    const name = folderName.trim();
    if (!activePath) {
      showToast("Directory unavailable", "error");
      return;
    }
    if (!name) {
      showToast("Folder name required", "error");
      return;
    }
    setCreating(true);
    try {
      await api.createDirectory(joinPath(activePath, name));
      setFolderName("");
      setCreateDialogOpen(false);
      showToast("Folder created", "success");
      await loadCurrentDirectory(currentPath);
    } catch (e) {
      showToast(`Create failed: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await api.uploadFile(joinPath(activePath, file.name), file, true);
      }
      showToast(`${files.length} file${files.length === 1 ? "" : "s"} uploaded`, "success");
      await loadCurrentDirectory(currentPath);
    } catch (e) {
      showToast(`Upload failed: ${e}`, "error");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDragEnter = (event: ReactDragEvent<HTMLElement>) => {
    if (!canUpload || !transferHasFiles(event)) return;
    event.preventDefault();
    dragDepthRef.current += 1;
    setDraggingFiles(true);
  };

  const handleDragOver = (event: ReactDragEvent<HTMLElement>) => {
    if (!canUpload || !transferHasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  };

  const handleDragLeave = (event: ReactDragEvent<HTMLElement>) => {
    if (!canUpload || !transferHasFiles(event)) return;
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) setDraggingFiles(false);
  };

  const handleDrop = (event: ReactDragEvent<HTMLElement>) => {
    if (!canUpload) return;
    event.preventDefault();
    dragDepthRef.current = 0;
    setDraggingFiles(false);
    void uploadFiles(event.dataTransfer.files);
  };

  const downloadFile = async (entry: ManagedFileEntry) => {
    if (entry.is_directory) return;
    try {
      const file = selectedFile?.path === entry.path ? selectedFile : await api.readFile(entry.path);
      downloadDataUrl(file.data_url, file.name);
    } catch (e) {
      showToast(`Download failed: ${e}`, "error");
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await api.deleteFile(pendingDelete.path, pendingDelete.is_directory);
      if (
        selectedEntry &&
        (selectedEntry.path === pendingDelete.path ||
          (pendingDelete.is_directory && isPathWithin(pendingDelete.path, selectedEntry.path)))
      ) {
        clearSelection();
      }
      showToast("Deleted", "success");
      setPendingDelete(null);
      await loadCurrentDirectory(currentPath);
    } catch (e) {
      showToast(`Delete failed: ${e}`, "error");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-4">
      <Toast toast={toast} />
      <PluginSlot name="files:top" />
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(event) => void uploadFiles(event.currentTarget.files)}
      />

      <div className="flex min-w-0 flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
        {canChangePath ? (
          <form
            className="flex min-w-0 flex-1 items-center gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              void goToPath();
            }}
          >
            <Input
              value={pathInput}
              onChange={(event) => setPathInput(event.target.value)}
              aria-label="Path"
              placeholder="Path"
              className="h-9 min-w-0 flex-1 font-mono"
            />
            <Button type="submit" size="sm" outlined className="uppercase">
              Go
            </Button>
          </form>
        ) : (
          <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto font-mono text-xs" aria-label="File path">
            {breadcrumbs.map((breadcrumb, index) => (
              <span key={breadcrumb.path} className="flex shrink-0 items-center gap-1">
                {index > 0 && <ChevronRight className="h-3.5 w-3.5 text-text-tertiary" />}
                <button
                  type="button"
                  onClick={() => openDirectory(breadcrumb.path)}
                  className="px-1 py-1 text-text-secondary hover:bg-background/40 hover:text-foreground"
                >
                  {breadcrumb.label}
                </button>
              </span>
            ))}
          </nav>
        )}
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={!canUpload}
            size="sm"
            outlined
            className="uppercase"
            prefix={uploading ? <Spinner /> : <Upload />}
          >
            Upload
          </Button>
          <Button
            type="button"
            onClick={() => setCreateDialogOpen(true)}
            disabled={!activePath}
            size="sm"
            outlined
            className="uppercase"
            prefix={<FolderPlus />}
          >
            Create
          </Button>
        </div>
      </div>

      <button
        type="button"
        onClick={() => canUpload && fileInputRef.current?.click()}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        disabled={!canUpload}
        aria-label="Upload files"
        className={`flex min-h-16 w-full min-w-0 items-center justify-between gap-4 border border-dashed px-4 py-3 text-left transition ${
          draggingFiles
            ? "border-primary bg-primary/10 text-foreground"
            : "border-border bg-background/20 text-text-secondary hover:border-text-tertiary hover:bg-background/35"
        } disabled:cursor-not-allowed disabled:opacity-60`}
      >
        <span className="flex min-w-0 items-center gap-3">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center border border-border bg-background/45 text-text-tertiary">
            {uploading ? <Spinner /> : <Upload className="h-4 w-4" />}
          </span>
          <span className="min-w-0">
            <span className="block text-sm font-semibold uppercase tracking-[0.08em] text-foreground">
              {uploading ? "Uploading" : draggingFiles ? "Release to upload" : "Drop files here"}
            </span>
            <span className="block truncate font-mono text-xs text-text-secondary" title={activePath}>
              {activePath || "Loading"}
            </span>
          </span>
        </span>
        <span className="hidden shrink-0 text-xs font-semibold uppercase tracking-[0.08em] text-text-tertiary sm:block">
          Choose files
        </span>
      </button>

      {error && (
        <div className="border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid min-h-[34rem] min-w-0 gap-4 lg:grid-cols-[17rem_minmax(0,1fr)]">
        <Card className="min-w-0 overflow-hidden">
          <CardContent className="flex h-full min-h-64 flex-col p-0">
            <div className="border-b border-border px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-text-tertiary">
              File tree
            </div>
            <div className="min-h-0 flex-1 overflow-auto py-1">
              {treeRoot ? (
                <FileTree
                  path={treeRoot}
                  label={pathName(treeRoot)}
                  depth={0}
                  currentPath={activePath}
                  selectedPath={selectedEntry?.path ?? null}
                  entriesByDirectory={entriesByDirectory}
                  expandedDirectories={expandedDirectories}
                  loadingDirectories={loadingDirectories}
                  onToggle={toggleDirectory}
                  onOpenDirectory={openDirectory}
                  onOpenFile={(entry) => void openFile(entry)}
                />
              ) : (
                <div className="flex items-center justify-center gap-2 py-10 text-sm text-text-secondary">
                  <Spinner /> Loading tree...
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="min-w-0 overflow-hidden">
          <CardContent className="min-w-0 p-0">
            {selectedEntry && selectedKind ? (
              <>
                <div className="flex min-w-0 flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <Button ghost size="icon" type="button" onClick={clearSelection} aria-label="Back to directory">
                      <ArrowLeft />
                    </Button>
                    {selectedKind === "markdown" ? (
                      <FileText className="h-4 w-4 shrink-0 text-primary" />
                    ) : selectedKind === "image" ? (
                      <ImageIcon className="h-4 w-4 shrink-0 text-text-tertiary" />
                    ) : (
                      <FileIcon className="h-4 w-4 shrink-0 text-text-tertiary" />
                    )}
                    <div className="min-w-0">
                      <div className="truncate font-mono text-sm font-medium" title={selectedEntry.path}>
                        {selectedEntry.name}
                      </div>
                      <div className="text-xs text-text-tertiary">{formatBytes(selectedEntry.size)}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {selectedKind === "markdown" && (
                      <div className="mr-1 flex border border-border">
                        <button
                          type="button"
                          onClick={() => setPreviewMode("rendered")}
                          className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold uppercase ${
                            previewMode === "rendered" ? "bg-secondary text-foreground" : "text-text-secondary"
                          }`}
                        >
                          <Eye className="h-3.5 w-3.5" /> Preview
                        </button>
                        <button
                          type="button"
                          onClick={() => setPreviewMode("source")}
                          className={`flex items-center gap-1.5 border-l border-border px-2.5 py-1.5 text-xs font-semibold uppercase ${
                            previewMode === "source" ? "bg-secondary text-foreground" : "text-text-secondary"
                          }`}
                        >
                          <Code2 className="h-3.5 w-3.5" /> Source
                        </button>
                      </div>
                    )}
                    <Button
                      ghost
                      size="icon"
                      type="button"
                      onClick={() => void downloadFile(selectedEntry)}
                      aria-label={`Download ${selectedEntry.name}`}
                    >
                      <Download />
                    </Button>
                    <Button
                      ghost
                      size="icon"
                      type="button"
                      onClick={() => setPendingDelete(selectedEntry)}
                      aria-label={`Delete ${selectedEntry.name}`}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 />
                    </Button>
                  </div>
                </div>
                <FilePreview
                  entry={selectedEntry}
                  kind={selectedKind}
                  file={selectedFile}
                  text={previewText}
                  mode={previewMode}
                  loading={previewLoading}
                  error={previewError}
                />
              </>
            ) : (
              <>
                <div className="flex items-center justify-between border-b border-border px-4 py-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <FolderOpen className="h-4 w-4 shrink-0 text-warning" />
                    <span className="truncate font-mono text-sm" title={activePath}>{pathName(activePath)}</span>
                  </div>
                  <Badge tone="outline" className="text-xs">
                    {listing?.entries.length ?? 0} items
                  </Badge>
                </div>
                <div className="overflow-x-auto">
                  <div className="grid min-w-[42rem] grid-cols-[minmax(12rem,1fr)_7rem_10rem_5.5rem] items-center gap-3 border-b border-border px-4 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-text-tertiary">
                    <span>Name</span>
                    <span>Size</span>
                    <span>Modified</span>
                    <span className="text-right">Actions</span>
                  </div>

                  {listing?.parent && (
                    <button
                      type="button"
                      onClick={() => openDirectory(listing.parent ?? activePath)}
                      className="grid w-full min-w-[42rem] grid-cols-[minmax(12rem,1fr)_7rem_10rem_5.5rem] items-center gap-3 border-b border-border/60 px-4 py-2 text-left text-sm transition hover:bg-background/40"
                    >
                      <span className="flex min-w-0 items-center gap-2 font-mono text-text-secondary">
                        <ArrowUp className="h-4 w-4 shrink-0 text-text-tertiary" /> ..
                      </span>
                      <span />
                      <span />
                      <span />
                    </button>
                  )}

                  {loading && !listing ? (
                    <div className="flex items-center justify-center gap-2 py-12 text-sm text-text-secondary">
                      <Spinner /> Loading files...
                    </div>
                  ) : listing && listing.entries.length === 0 ? (
                    <div className="py-12 text-center text-sm text-text-secondary">No files</div>
                  ) : (
                    listing?.entries.map((entry) => (
                      <div
                        key={entry.path}
                        className="grid min-w-[42rem] grid-cols-[minmax(12rem,1fr)_7rem_10rem_5.5rem] items-center gap-3 border-b border-border/60 px-4 py-2 text-sm last:border-b-0 hover:bg-background/35"
                      >
                        <button
                          type="button"
                          onClick={() => entry.is_directory ? openDirectory(entry.path) : void openFile(entry)}
                          className="flex min-w-0 items-center gap-2 text-left font-mono text-foreground"
                        >
                          {entry.is_directory ? (
                            <Folder className="h-4 w-4 shrink-0 text-warning" />
                          ) : previewKind(entry.name, entry.mime_type, entry.size ?? 0) === "markdown" ? (
                            <FileText className="h-4 w-4 shrink-0 text-primary" />
                          ) : (
                            <FileIcon className="h-4 w-4 shrink-0 text-text-tertiary" />
                          )}
                          <span className="truncate">{entry.name}</span>
                        </button>
                        <span className="text-xs tabular-nums text-text-secondary">{formatBytes(entry.size)}</span>
                        <span className="truncate text-xs text-text-secondary">
                          {Number.isFinite(entry.mtime) ? DATE_FORMAT.format(entry.mtime * 1000) : "-"}
                        </span>
                        <span className="flex justify-end gap-1">
                          {entry.is_directory ? (
                            <Button ghost size="icon" type="button" onClick={() => openDirectory(entry.path)} aria-label={`Open ${entry.name}`}>
                              <FolderOpen />
                            </Button>
                          ) : (
                            <Button ghost size="icon" type="button" onClick={() => void downloadFile(entry)} aria-label={`Download ${entry.name}`}>
                              <Download />
                            </Button>
                          )}
                          <Button
                            ghost
                            size="icon"
                            type="button"
                            onClick={() => setPendingDelete(entry)}
                            aria-label={`Delete ${entry.name}`}
                            className="text-destructive hover:text-destructive"
                          >
                            <Trash2 />
                          </Button>
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <PluginSlot name="files:bottom" />

      <Dialog
        open={createDialogOpen}
        onOpenChange={(open) => {
          if (creating) return;
          setCreateDialogOpen(open);
          if (!open) setFolderName("");
        }}
      >
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Create folder</DialogTitle>
            <DialogDescription>Target: {activePath || "Loading"}</DialogDescription>
          </DialogHeader>
          <div className="p-4">
            <Input
              autoFocus
              value={folderName}
              onChange={(event) => setFolderName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void createDirectory();
              }}
              placeholder="Folder name"
              disabled={creating}
            />
          </div>
          <DialogFooter>
            <Button
              type="button"
              outlined
              onClick={() => {
                setCreateDialogOpen(false);
                setFolderName("");
              }}
              disabled={creating}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void createDirectory()}
              disabled={creating}
              prefix={creating ? <Spinner /> : <FolderPlus />}
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <DeleteConfirmDialog
        open={Boolean(pendingDelete)}
        loading={deleting}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => void confirmDelete()}
        title={pendingDelete ? `Delete ${pendingDelete.name}?` : "Delete item?"}
        description={
          pendingDelete?.is_directory
            ? "This removes the folder and everything inside it."
            : "This removes the file."
        }
      />
    </div>
  );
}
