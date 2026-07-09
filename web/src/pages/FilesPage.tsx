import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent as ReactDragEvent,
} from "react";
import {
  ArrowUp,
  Download,
  FileIcon,
  Folder,
  FolderOpen,
  FolderPlus,
  RefreshCw,
  Search,
  Trash2,
  Upload,
  X,
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
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type { ManagedFileEntry, ManagedFilesResponse } from "@/lib/api";
import {
  displayPath,
  extractManagedFileErrorPath,
  filterManagedEntries,
  formatBytes,
  joinPath,
  parentPathOf,
} from "@/lib/files-ui";
import { PluginSlot } from "@/plugins";

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

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
  const [currentPath, setCurrentPath] = useState<string | undefined>(undefined);
  const [pathInput, setPathInput] = useState("");
  const [listing, setListing] = useState<ManagedFilesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [draggingFiles, setDraggingFiles] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [folderName, setFolderName] = useState("");
  const [pendingDelete, setPendingDelete] = useState<ManagedFileEntry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [failedPath, setFailedPath] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const activePath = listing?.path ?? currentPath ?? pathInput.trim();
  const canChangePath = listing?.can_change_path ?? true;
  const canUseActiveDirectory = Boolean(listing?.path);
  const canUpload = canUseActiveDirectory && !uploading;
  const headerPath = displayPath(listing?.locked_root ?? listing?.path ?? currentPath ?? failedPath);
  const filteredEntries = useMemo(
    () => filterManagedEntries(listing?.entries ?? [], search),
    [listing?.entries, search],
  );
  const recoveryPath = parentPathOf(failedPath ?? pathInput);

  const load = useCallback(
    async (path = currentPath) => {
      setLoading(true);
      setError(null);
      setFailedPath(null);
      try {
        const result = await api.listFiles(path);
        setListing(result);
        setCurrentPath(result.path);
        setPathInput(result.path);
        setSearch("");
      } catch (e) {
        const message = String(e);
        const extractedPath = extractManagedFileErrorPath(message);
        setListing(null);
        setError(message);
        setFailedPath(extractedPath);
        if (extractedPath && !pathInput.trim()) {
          setPathInput(extractedPath);
        }
      } finally {
        setLoading(false);
      }
    },
    [currentPath, pathInput],
  );

  useEffect(() => {
    // Existing dashboard data pages fetch from effects; keep this local and explicit
    // until the shared lint profile is updated for async page loaders.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load(currentPath);
  }, [currentPath]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setAfterTitle(
      <Badge tone="outline" className="max-w-[22rem] truncate text-xs" title={headerPath}>
        {headerPath}
      </Badge>,
    );
    setEnd(
      <div className="flex items-center gap-2">
        <Button
          ghost
          size="icon"
          type="button"
          onClick={() => void load()}
          disabled={loading}
          aria-label="Refresh files"
        >
          {loading ? <Spinner /> : <RefreshCw />}
        </Button>
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [headerPath, load, loading, setAfterTitle, setEnd]);

  const openDirectory = (entry: ManagedFileEntry) => {
    if (entry.is_directory) {
      setCurrentPath(entry.path);
    }
  };

  const goToPath = async () => {
    const nextPath = pathInput.trim();
    if (!nextPath) {
      showToast("Path required", "error");
      return;
    }
    await load(nextPath);
  };

  const createDirectory = async () => {
    const name = folderName.trim();
    if (!canUseActiveDirectory) {
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
      await load();
    } catch (e) {
      showToast(`Create failed: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const uploadFiles = async (files: FileList | null) => {
    if (!files?.length || !canUseActiveDirectory) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await api.uploadFile(joinPath(activePath, file.name), file, true);
      }
      showToast(`${files.length} file${files.length === 1 ? "" : "s"} uploaded`, "success");
      await load();
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
    if (dragDepthRef.current === 0) {
      setDraggingFiles(false);
    }
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
      const file = await api.readFile(entry.path);
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
      showToast("Deleted", "success");
      setPendingDelete(null);
      await load();
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

      <div className="grid min-w-0 gap-3 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center">
        <div className="grid min-w-0 gap-2 md:grid-cols-[minmax(0,1fr)_minmax(12rem,18rem)] md:items-center">
          {canChangePath ? (
            <form
              className="flex min-w-0 items-center gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                void goToPath();
              }}
            >
              <Input
                value={pathInput}
                onChange={(event) => setPathInput(event.target.value)}
                aria-label="Path"
                placeholder="Absolute path"
                className="h-9 min-w-0 flex-1 font-mono"
              />
              <Button type="submit" size="sm" outlined className="uppercase">
                Go
              </Button>
            </form>
          ) : (
            <div className="min-w-0 truncate font-mono text-sm text-text-secondary" title={activePath}>
              {activePath}
            </div>
          )}

          <label className="flex min-w-0 items-center gap-2 border border-border bg-background/20 px-3 py-2">
            <Search className="h-4 w-4 shrink-0 text-text-tertiary" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              aria-label="Search files"
              placeholder="Search files"
              className="h-5 min-w-0 border-0 bg-transparent p-0 text-sm shadow-none focus-visible:ring-0"
              disabled={!listing}
            />
            {search ? (
              <Button
                ghost
                size="icon"
                type="button"
                onClick={() => setSearch("")}
                aria-label="Clear file search"
                className="h-5 w-5 shrink-0"
              >
                <X className="h-3 w-3" />
              </Button>
            ) : null}
          </label>
        </div>

        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Button
            type="button"
            onClick={() => void load(pathInput.trim() || currentPath)}
            disabled={loading}
            size="sm"
            outlined
            className="uppercase"
            prefix={loading ? <Spinner /> : <RefreshCw />}
          >
            Refresh
          </Button>
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
            disabled={!canUseActiveDirectory}
            size="sm"
            outlined
            className="uppercase"
            prefix={<FolderPlus />}
          >
            New folder
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
        className={`flex min-h-20 w-full min-w-0 items-center justify-between gap-4 border border-dashed px-4 py-3 text-left transition ${
          draggingFiles
            ? "border-primary bg-primary/10 text-foreground"
            : "border-border bg-background/20 text-text-secondary hover:border-text-tertiary hover:bg-background/35"
        } disabled:cursor-not-allowed disabled:opacity-60`}
      >
        <span className="flex min-w-0 items-center gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center border border-border bg-background/45 text-text-tertiary">
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

      <Card className="min-w-0 max-w-full overflow-hidden">
        <CardContent className="p-0">
          {error && (
            <div className="grid gap-3 border-b border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
              <div className="break-words font-mono text-xs leading-5">{error}</div>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="sm"
                  outlined
                  onClick={() => void load(pathInput.trim() || currentPath)}
                  disabled={loading}
                  prefix={loading ? <Spinner /> : <RefreshCw />}
                >
                  Retry
                </Button>
                {recoveryPath ? (
                  <Button
                    type="button"
                    size="sm"
                    outlined
                    onClick={() => void load(recoveryPath)}
                  >
                    Open parent
                  </Button>
                ) : null}
              </div>
            </div>
          )}

          <div className="hidden md:block md:overflow-x-auto">
            <div className="grid min-w-[42rem] grid-cols-[minmax(12rem,1fr)_7rem_10rem_5.5rem] items-center gap-3 border-b border-border px-4 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-text-tertiary">
              <span>Name</span>
              <span>Size</span>
              <span>Modified</span>
              <span className="text-right">Actions</span>
            </div>

            {listing?.parent && (
              <button
                type="button"
                onClick={() => setCurrentPath(listing.parent ?? undefined)}
                className="grid w-full min-w-[42rem] grid-cols-[minmax(12rem,1fr)_7rem_10rem_5.5rem] items-center gap-3 border-b border-border/60 px-4 py-2 text-left text-sm transition hover:bg-background/40"
              >
                <span className="flex min-w-0 items-center gap-2 font-mono text-text-secondary">
                  <ArrowUp className="h-4 w-4 shrink-0 text-text-tertiary" />
                  ..
                </span>
                <span />
                <span />
                <span />
              </button>
            )}

            {loading && !listing ? (
              <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
                <Spinner />
                Loading files...
              </div>
            ) : listing && listing.entries.length === 0 ? (
              <div className="py-12 text-center text-sm text-muted-foreground">No files</div>
            ) : listing && filteredEntries.length === 0 ? (
              <div className="py-12 text-center text-sm text-muted-foreground">No matching files</div>
            ) : (
              filteredEntries.map((entry) => (
                <div
                  key={entry.path}
                  className="grid min-w-[42rem] grid-cols-[minmax(12rem,1fr)_7rem_10rem_5.5rem] items-center gap-3 border-b border-border/60 px-4 py-2 text-sm last:border-b-0 hover:bg-background/35"
                >
                  <button
                    type="button"
                    onClick={() => (entry.is_directory ? openDirectory(entry) : void downloadFile(entry))}
                    className="flex min-w-0 items-center gap-2 text-left font-mono text-foreground"
                  >
                    {entry.is_directory ? (
                      <Folder className="h-4 w-4 shrink-0 text-warning" />
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
                      <Button
                        ghost
                        size="icon"
                        type="button"
                        onClick={() => openDirectory(entry)}
                        aria-label={`Open ${entry.name}`}
                      >
                        <FolderOpen />
                      </Button>
                    ) : (
                      <Button
                        ghost
                        size="icon"
                        type="button"
                        onClick={() => void downloadFile(entry)}
                        aria-label={`Download ${entry.name}`}
                      >
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

          <div className="md:hidden">
            {listing?.parent && (
              <button
                type="button"
                onClick={() => setCurrentPath(listing.parent ?? undefined)}
                className="flex w-full items-center gap-2 border-b border-border/60 px-4 py-3 text-left font-mono text-sm text-text-secondary"
              >
                <ArrowUp className="h-4 w-4 shrink-0 text-text-tertiary" />
                ..
              </button>
            )}

            {loading && !listing ? (
              <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
                <Spinner />
                Loading files...
              </div>
            ) : listing && listing.entries.length === 0 ? (
              <div className="py-12 text-center text-sm text-muted-foreground">No files</div>
            ) : listing && filteredEntries.length === 0 ? (
              <div className="py-12 text-center text-sm text-muted-foreground">No matching files</div>
            ) : (
              filteredEntries.map((entry) => (
                <div key={entry.path} className="grid gap-3 border-b border-border/60 px-4 py-3 last:border-b-0">
                  <button
                    type="button"
                    onClick={() => (entry.is_directory ? openDirectory(entry) : void downloadFile(entry))}
                    className="flex min-w-0 items-center gap-2 text-left font-mono text-sm text-foreground"
                  >
                    {entry.is_directory ? (
                      <Folder className="h-4 w-4 shrink-0 text-warning" />
                    ) : (
                      <FileIcon className="h-4 w-4 shrink-0 text-text-tertiary" />
                    )}
                    <span className="min-w-0 truncate">{entry.name}</span>
                  </button>
                  <div className="grid gap-1 text-xs text-text-secondary">
                    <span>{formatBytes(entry.size)}</span>
                    <span className="truncate">
                      {Number.isFinite(entry.mtime) ? DATE_FORMAT.format(entry.mtime * 1000) : "-"}
                    </span>
                  </div>
                  <div className="flex flex-wrap justify-end gap-1">
                    {entry.is_directory ? (
                      <Button
                        ghost
                        size="sm"
                        type="button"
                        onClick={() => openDirectory(entry)}
                        aria-label={`Open ${entry.name}`}
                        prefix={<FolderOpen />}
                      >
                        Open
                      </Button>
                    ) : (
                      <Button
                        ghost
                        size="sm"
                        type="button"
                        onClick={() => void downloadFile(entry)}
                        aria-label={`Download ${entry.name}`}
                        prefix={<Download />}
                      >
                        Download
                      </Button>
                    )}
                    <Button
                      ghost
                      size="sm"
                      type="button"
                      onClick={() => setPendingDelete(entry)}
                      aria-label={`Delete ${entry.name}`}
                      className="text-destructive hover:text-destructive"
                      prefix={<Trash2 />}
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

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
            <DialogDescription>
              Target: {activePath || "Loading"}
            </DialogDescription>
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
