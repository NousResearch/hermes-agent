import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Download,
  File,
  Folder,
  FolderOpen,
  Home,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type FileBrowserEntry, type FileBrowserListing } from "@/lib/api";
import { cn } from "@/lib/utils";
import { PluginSlot } from "@/plugins";

function formatSize(size: number | null): string {
  if (size === null) return "—";
  if (size < 1024) return `${size} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = size / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function formatModified(timestamp: number): string {
  if (!Number.isFinite(timestamp)) return "—";
  return new Date(timestamp * 1000).toLocaleString();
}

function Breadcrumbs({ path, onNavigate }: { path: string; onNavigate: (path: string) => void }) {
  const parts = useMemo(() => path.split("/").filter(Boolean), [path]);
  let current = "";

  return (
    <nav aria-label="File browser path" className="flex min-w-0 flex-wrap items-center gap-1 text-sm text-text-secondary">
      <Button
        type="button"
        ghost
        className="h-7 px-2 text-text-secondary hover:text-midground"
        onClick={() => onNavigate("")}
      >
        <Home className="mr-1 h-3.5 w-3.5" />
        Files
      </Button>
      {parts.map((part) => {
        current = current ? `${current}/${part}` : part;
        const target = current;
        return (
          <span className="flex min-w-0 items-center gap-1" key={target}>
            <span className="text-text-tertiary">/</span>
            <Button
              type="button"
              ghost
              className="h-7 max-w-48 px-2 text-text-secondary hover:text-midground"
              onClick={() => onNavigate(target)}
              title={target}
            >
              <span className="truncate">{part}</span>
            </Button>
          </span>
        );
      })}
    </nav>
  );
}

export default function FileBrowserPage() {
  const [currentPath, setCurrentPath] = useState("");
  const [listing, setListing] = useState<FileBrowserListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadFiles = useCallback((path = currentPath) => {
    setLoading(true);
    setError(null);
    api.getFiles(path)
      .then((resp) => {
        setListing(resp);
        setCurrentPath(resp.path);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [currentPath]);

  useEffect(() => {
    loadFiles("");
    // Run only on mount; loadFiles intentionally depends on currentPath for refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openEntry = (entry: FileBrowserEntry) => {
    if (entry.type === "directory") {
      loadFiles(entry.path);
    }
  };

  const downloadEntry = async (entry: FileBrowserEntry) => {
    if (entry.type !== "file") return;
    setDownloading(entry.path);
    setError(null);
    try {
      const blob = await api.downloadFile(entry.path);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = entry.name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(String(err));
    } finally {
      setDownloading(null);
    }
  };

  const deleteEntry = async (entry: FileBrowserEntry) => {
    if (entry.type !== "file") return;
    if (!window.confirm(`Delete ${entry.name}? This cannot be undone.`)) return;
    setDeleting(entry.path);
    setError(null);
    try {
      await api.deleteFile(entry.path);
      loadFiles(currentPath);
    } catch (err) {
      setError(String(err));
    } finally {
      setDeleting(null);
    }
  };

  const uploadFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      await api.uploadDocuments(Array.from(files), currentPath || "uploads");
      loadFiles(currentPath);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(String(err));
    } finally {
      setUploading(false);
    }
  };

  const entries = listing?.entries ?? [];

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-4">
      <PluginSlot name="files:top" />

      <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5 text-midground" />
            <h2 className="font-mondwest text-display text-xl uppercase tracking-[0.12em] text-midground">
              File Browser
            </h2>
            <Badge tone="secondary" className="text-xs">
              {entries.length} {entries.length === 1 ? "item" : "items"}
            </Badge>
          </div>
          <p className="mt-1 text-sm text-text-secondary">
            Browse and download files from the dashboard file root.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => void uploadFiles(event.currentTarget.files)}
          />
          <Button
            type="button"
            ghost
            className="text-text-secondary hover:text-midground"
            onClick={() => fileInputRef.current?.click()}
            disabled={loading || uploading}
            title={`Upload into ${currentPath || "uploads"}`}
          >
            {uploading ? <Spinner /> : <Upload className="h-4 w-4" />}
            <span className="ml-2">Upload here</span>
          </Button>
          <Button
            type="button"
            ghost
            className="text-text-secondary hover:text-midground"
            onClick={() => loadFiles(currentPath)}
            disabled={loading || uploading}
          >
            {loading ? <Spinner /> : <RefreshCw className="h-4 w-4" />}
            <span className="ml-2">Refresh</span>
          </Button>
        </div>
      </div>

      <Card className="min-w-0 max-w-full overflow-hidden">
        <CardHeader className="px-4 py-3">
          <CardTitle className="flex min-w-0 items-center justify-between gap-3 text-sm">
            <Breadcrumbs path={listing?.path ?? currentPath} onNavigate={loadFiles} />
            {loading && <Spinner className="shrink-0" />}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {error && (
            <div className="border-b border-destructive/20 bg-destructive/10 p-3">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead className="border-b border-current/10 text-xs uppercase tracking-[0.12em] text-text-tertiary">
                <tr>
                  <th className="px-4 py-2 font-mondwest text-display">Name</th>
                  <th className="px-4 py-2 font-mondwest text-display">Type</th>
                  <th className="px-4 py-2 font-mondwest text-display">Size</th>
                  <th className="px-4 py-2 font-mondwest text-display">Modified</th>
                  <th className="px-4 py-2 text-right font-mondwest text-display">Action</th>
                </tr>
              </thead>
              <tbody>
                {listing?.parent !== null && listing && (
                  <tr className="border-b border-current/10 text-text-secondary hover:bg-midground/5">
                    <td className="px-4 py-2" colSpan={5}>
                      <button
                        type="button"
                        className="flex items-center gap-2 text-left hover:text-midground"
                        onClick={() => loadFiles(listing.parent ?? "")}
                      >
                        <Folder className="h-4 w-4" />
                        ..
                      </button>
                    </td>
                  </tr>
                )}

                {entries.map((entry) => {
                  const isDirectory = entry.type === "directory";
                  const isDownloading = downloading === entry.path;
                  const isDeleting = deleting === entry.path;
                  return (
                    <tr
                      key={entry.path}
                      className="border-b border-current/10 text-text-secondary hover:bg-midground/5"
                    >
                      <td className="max-w-[24rem] px-4 py-2">
                        <button
                          type="button"
                          className={cn(
                            "flex min-w-0 items-center gap-2 text-left",
                            isDirectory ? "cursor-pointer hover:text-midground" : "cursor-default",
                          )}
                          onClick={() => openEntry(entry)}
                        >
                          {isDirectory ? (
                            <Folder className="h-4 w-4 shrink-0 text-midground" />
                          ) : (
                            <File className="h-4 w-4 shrink-0" />
                          )}
                          <span className="truncate" title={entry.name}>{entry.name}</span>
                        </button>
                      </td>
                      <td className="px-4 py-2 capitalize">{entry.type}</td>
                      <td className="px-4 py-2 tabular-nums">{formatSize(entry.size)}</td>
                      <td className="px-4 py-2 tabular-nums">{formatModified(entry.modified_at)}</td>
                      <td className="px-4 py-2 text-right">
                        {!isDirectory && (
                          <div className="flex justify-end gap-1">
                            <Button
                              type="button"
                              ghost
                              className="h-8 px-2 text-text-secondary hover:text-midground"
                              onClick={() => void downloadEntry(entry)}
                              disabled={isDownloading || isDeleting}
                              title={`Download ${entry.name}`}
                            >
                              {isDownloading ? <Spinner /> : <Download className="h-4 w-4" />}
                            </Button>
                            <Button
                              type="button"
                              ghost
                              className="h-8 px-2 text-text-secondary hover:text-destructive"
                              onClick={() => void deleteEntry(entry)}
                              disabled={isDownloading || isDeleting}
                              title={`Delete ${entry.name}`}
                            >
                              {isDeleting ? <Spinner /> : <Trash2 className="h-4 w-4" />}
                              <span className="sr-only">Delete</span>
                            </Button>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}

                {!loading && entries.length === 0 && (
                  <tr>
                    <td className="px-4 py-8 text-center text-text-tertiary" colSpan={5}>
                      This folder is empty.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <PluginSlot name="files:bottom" />
    </div>
  );
}
