import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Download,
  FileIcon,
  Folder,
  FolderOpen,
  FolderPlus,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type { ManagedFileEntry, ManagedFilesResponse } from "@/lib/api";
import { PluginSlot } from "@/plugins";

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function joinPath(base: string, name: string): string {
  const cleanName = name.trim().replace(/^[\\/]+/, "");
  if (!cleanName) return base;
  const separator = base.includes("\\") && !base.includes("/") ? "\\" : "/";
  if (!base || base.endsWith("/") || base.endsWith("\\")) return `${base}${cleanName}`;
  return `${base}${separator}${cleanName}`;
}

function formatBytes(size: number | null): string {
  if (size === null) return "-";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function readAsDataUrl(file: globalThis.File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("Could not read file"));
    });
    reader.addEventListener("error", () => reject(reader.error ?? new Error("Could not read file")));
    reader.readAsDataURL(file);
  });
}

function downloadDataUrl(dataUrl: string, name: string) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = name || "download";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function displayPath(path: string | null | undefined): string {
  return path?.trim() || "Files";
}

export default function FilesPage() {
  const { toast, showToast } = useToast();
  const { setAfterTitle, setEnd } = usePageHeader();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [currentPath, setCurrentPath] = useState<string | undefined>(undefined);
  const [pathInput, setPathInput] = useState("");
  const [listing, setListing] = useState<ManagedFilesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [folderName, setFolderName] = useState("");
  const [pendingDelete, setPendingDelete] = useState<ManagedFileEntry | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activePath = listing?.path ?? currentPath ?? "";
  const canChangePath = listing?.can_change_path ?? false;
  const headerPath = displayPath(listing?.locked_root ?? listing?.path ?? currentPath);

  const load = useCallback(
    async (path = currentPath) => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.listFiles(path);
        setListing(result);
        setCurrentPath(result.path);
        setPathInput(result.path);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [currentPath],
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
    if (!name) {
      showToast("Folder name required", "error");
      return;
    }
    setCreating(true);
    try {
      await api.createDirectory(joinPath(activePath, name));
      setFolderName("");
      showToast("Folder created", "success");
      await load();
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
        const dataUrl = await readAsDataUrl(file);
        await api.uploadFile(joinPath(activePath, file.name), dataUrl, true);
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
          <div className="min-w-0 truncate font-mono text-sm text-text-secondary" title={activePath}>
            {activePath}
          </div>
        )}
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading || !activePath}
            size="sm"
            outlined
            className="uppercase"
            prefix={uploading ? <Spinner /> : <Upload />}
          >
            Upload
          </Button>
          <Input
            value={folderName}
            onChange={(event) => setFolderName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void createDirectory();
            }}
            placeholder="New folder"
            className="h-9 w-44"
          />
          <Button
            type="button"
            onClick={() => void createDirectory()}
            disabled={creating || !activePath}
            size="sm"
            outlined
            className="uppercase"
            prefix={creating ? <Spinner /> : <FolderPlus />}
          >
            Create
          </Button>
        </div>
      </div>

      <Card className="min-w-0 max-w-full overflow-hidden">
        <CardContent className="overflow-x-auto p-0">
          {error && (
            <div className="border-b border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
              {error}
            </div>
          )}

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
          ) : (
            listing?.entries.map((entry) => (
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
        </CardContent>
      </Card>

      <PluginSlot name="files:bottom" />

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
