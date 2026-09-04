import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Archive,
  Check,
  Clipboard,
  Download,
  Eye,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { copyTextToClipboard } from "@/lib/clipboard";
import { artifactDownloadName } from "@/lib/artifact-detect";
import { sandboxedArtifactDocument } from "@/lib/artifact-preview";
import {
  artifactSessionIds,
  filterStoredArtifacts,
  sortStoredArtifacts,
  type ArtifactKindFilter,
} from "@/lib/artifact-library";
import {
  ARTIFACT_STORAGE_CHANGE_EVENT,
  artifactStorageBytes,
  clearStoredArtifacts,
  getArtifactStorage,
  MAX_PERSISTED_ARTIFACT_STORAGE_BYTES,
  getArtifactStorageKey,
  readStoredArtifacts,
  setArtifactPinnedWithResult,
  type ArtifactStorageFailureReason,
  type StoredArtifact,
} from "@/lib/artifact-storage";

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function artifactBytes(artifact: StoredArtifact): number {
  return new TextEncoder().encode(artifact.code).byteLength;
}

function formatArtifactDate(timestamp: number): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? "Unknown date" : DATE_FORMAT.format(date);
}

function failureMessage(reason: ArtifactStorageFailureReason): string {
  switch (reason) {
    case "artifact-too-large":
      return "This artifact is larger than the 200 KB per-artifact limit.";
    case "storage-limit":
      return "Pinned artifact storage is full. Remove an artifact before adding another.";
    case "storage-quota":
      return "The browser declined this storage change. Free browser storage and try again.";
    case "storage-unavailable":
      return "Browser storage is unavailable in this session.";
  }
}

function downloadArtifact(artifact: StoredArtifact): void {
  const blob = new Blob([artifact.code], {
    type: artifact.kind === "html"
      ? "text/html;charset=utf-8"
      : artifact.kind === "svg"
        ? "image/svg+xml;charset=utf-8"
        : "text/plain;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = artifactDownloadName(artifact.kind, artifact.language, artifact.title);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function ArtifactPreview({ artifact, onClose }: { artifact: StoredArtifact; onClose: () => void }) {
  const isDocument = artifact.kind === "html" || artifact.kind === "svg";
  const modalRef = useModalBehavior({ open: true, onClose });
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`${artifact.title} preview`}
      className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4"
    >
      <div ref={modalRef} className="flex max-h-[min(48rem,calc(100dvh-2rem))] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-border bg-card text-card-foreground shadow-xl">
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h2 className="truncate font-expanded text-base">{artifact.title}</h2>
            <p className="text-xs text-text-secondary">
              {artifact.kind.toUpperCase()} · {artifact.language || "plain text"} · sandboxed preview
            </p>
          </div>
          <Button type="button" ghost size="icon" aria-label="Close artifact preview" onClick={onClose}>
            <X />
          </Button>
        </div>
        <div className="min-h-0 overflow-auto p-4">
          {isDocument ? (
            <iframe
              title={`${artifact.title} sandboxed preview`}
              sandbox=""
              referrerPolicy="no-referrer"
              srcDoc={sandboxedArtifactDocument(artifact.code)}
              className="h-[min(34rem,65dvh)] w-full rounded border border-border bg-white"
            />
          ) : (
            <pre className="max-h-[min(34rem,65dvh)] overflow-auto whitespace-pre-wrap rounded border border-border bg-background p-4 font-mono text-xs text-foreground">
              {artifact.code}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function ClearArtifactsDialog({
  count,
  onCancel,
  onConfirm,
}: {
  count: number;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const modalRef = useModalBehavior({ open: true, onClose: onCancel });
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Clear pinned artifacts"
      className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4"
    >
      <div ref={modalRef} className="w-full max-w-md rounded-lg border border-border bg-card p-5 text-card-foreground shadow-xl">
        <h2 className="font-expanded text-base">Clear pinned artifacts?</h2>
        <p className="mt-2 text-sm text-text-secondary">
          This removes {count} locally pinned artifact{count === 1 ? "" : "s"} from this profile. Chat history is not changed.
        </p>
        <div className="mt-5 flex justify-end gap-2">
          <Button type="button" ghost onClick={onCancel}>Cancel</Button>
          <Button type="button" destructive onClick={onConfirm}>Clear artifacts</Button>
        </div>
      </div>
    </div>
  );
}

export default function ArtifactsPage() {
  const { profile, currentProfile } = useProfileScope();
  const artifactProfile = profile || currentProfile || "default";
  const { setAfterTitle, setEnd } = usePageHeader();
  const storage = useMemo(() => getArtifactStorage(), []);
  const [artifacts, setArtifacts] = useState<StoredArtifact[]>([]);
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<ArtifactKindFilter>("all");
  const [sessionId, setSessionId] = useState("all");
  const [preview, setPreview] = useState<StoredArtifact | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [clearOpen, setClearOpen] = useState(false);

  const storageKey = getArtifactStorageKey(artifactProfile, storage);
  const refresh = useCallback(() => {
    setArtifacts(sortStoredArtifacts(readStoredArtifacts(storage, artifactProfile)));
  }, [artifactProfile, storage]);

  useEffect(() => {
    refresh();
    const onStorage = (event: StorageEvent) => {
      if (event.key === storageKey) refresh();
    };
    const onArtifactStorageChange = (event: Event) => {
      const key = (event as CustomEvent<{ key?: string }>).detail?.key;
      if (!key || key === storageKey) refresh();
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener(ARTIFACT_STORAGE_CHANGE_EVENT, onArtifactStorageChange);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(ARTIFACT_STORAGE_CHANGE_EVENT, onArtifactStorageChange);
    };
  }, [refresh, storageKey]);

  useEffect(() => {
    setAfterTitle(
      <Badge tone="outline" className="max-w-[16rem] truncate text-xs">
        {artifactProfile}
      </Badge>,
    );
    setEnd(
      <Button type="button" ghost size="icon" aria-label="Refresh artifacts" onClick={refresh}>
        <RefreshCw />
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [artifactProfile, refresh, setAfterTitle, setEnd]);

  const visibleArtifacts = useMemo(
    () => filterStoredArtifacts(artifacts, { query, kind, sessionId }),
    [artifacts, kind, query, sessionId],
  );
  const sessions = useMemo(() => artifactSessionIds(artifacts), [artifacts]);
  const usedBytes = useMemo(() => artifactStorageBytes(artifacts), [artifacts]);

  const handleCopy = useCallback(async (artifact: StoredArtifact) => {
    setActionError(null);
    try {
      const copied = await copyTextToClipboard(artifact.code);
      if (!copied) {
        setActionError("Copy failed. The browser did not grant clipboard access.");
        return;
      }
      setCopiedId(artifact.id);
      window.setTimeout(() => setCopiedId((current) => current === artifact.id ? null : current), 1600);
    } catch {
      setActionError("Copy failed. The browser did not grant clipboard access.");
    }
  }, []);

  const handleUnpin = useCallback((artifact: StoredArtifact) => {
    const result = setArtifactPinnedWithResult(storage, artifact, false, artifactProfile);
    if (!result.ok) {
      setActionError(failureMessage(result.reason));
      return;
    }
    setActionError(null);
    setArtifacts(sortStoredArtifacts(result.artifacts));
    if (preview?.id === artifact.id) setPreview(null);
  }, [artifactProfile, preview?.id, storage]);

  const handleClear = useCallback(() => {
    const result = clearStoredArtifacts(storage, artifactProfile);
    if (!result.ok) {
      setActionError(failureMessage(result.reason));
      return;
    }
    setArtifacts([]);
    setClearOpen(false);
    setActionError(null);
  }, [artifactProfile, storage]);

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-4">
      {preview && <ArtifactPreview artifact={preview} onClose={() => setPreview(null)} />}
      {clearOpen && (
        <ClearArtifactsDialog count={artifacts.length} onCancel={() => setClearOpen(false)} onConfirm={handleClear} />
      )}

      <div className="flex min-w-0 flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
        <div className="min-w-0 flex-1">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-tertiary" aria-hidden />
            <Input
              type="search"
              aria-label="Search pinned artifacts"
              placeholder="Search title, code, language, or session"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              className="h-10 pl-9"
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-text-secondary">
            <span>Kind</span>
            <select
              aria-label="Filter artifacts by kind"
              value={kind}
              onChange={(event) => setKind(event.target.value as ArtifactKindFilter)}
              className="h-10 rounded border border-border bg-background px-2 text-xs text-foreground"
            >
              <option value="all">All kinds</option>
              <option value="code">Code</option>
              <option value="html">HTML</option>
              <option value="svg">SVG</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-xs text-text-secondary">
            <span>Session</span>
            <select
              aria-label="Filter artifacts by session"
              value={sessionId}
              onChange={(event) => setSessionId(event.target.value)}
              className="h-10 max-w-[14rem] rounded border border-border bg-background px-2 text-xs text-foreground"
            >
              <option value="all">All sessions</option>
              {sessions.map((id) => <option key={id} value={id}>{id}</option>)}
            </select>
          </label>
          <Button
            type="button"
            ghost
            onClick={() => setClearOpen(true)}
            disabled={artifacts.length === 0}
            aria-label="Clear all pinned artifacts"
          >
            <Trash2 />
            Clear all
          </Button>
        </div>
      </div>

      {actionError && <p role="alert" className="rounded border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning">{actionError}</p>}
      {!storage && <p role="alert" className="rounded border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning">Browser storage is unavailable; pinned artifacts cannot be loaded or saved.</p>}

      <div className="flex flex-wrap items-center gap-2 text-xs text-text-secondary" aria-live="polite">
        <Badge tone="outline"><Archive /> {visibleArtifacts.length} shown</Badge>
        <span>{artifacts.length} pinned</span>
        <span>·</span>
        <span>{formatBytes(usedBytes)} / {formatBytes(MAX_PERSISTED_ARTIFACT_STORAGE_BYTES)} used</span>
      </div>

      {visibleArtifacts.length === 0 ? (
        <Card>
          <CardContent className="flex min-h-48 flex-col items-center justify-center gap-2 text-center">
            <Archive className="h-8 w-8 text-text-tertiary" aria-hidden />
            <p className="font-expanded text-base">{artifacts.length === 0 ? "No pinned artifacts yet" : "No artifacts match these filters"}</p>
            <p className="max-w-md text-sm text-text-secondary">
              {artifacts.length === 0
                ? "Pin a generated code, HTML, or SVG artifact from Chat to keep it available here."
                : "Try a different search term, kind, or source session."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid min-w-0 gap-3 lg:grid-cols-2">
          {visibleArtifacts.map((artifact) => (
            <Card key={artifact.id} data-testid="artifact-library-item" className="min-w-0 overflow-hidden">
              <CardHeader className="gap-2 pb-3">
                <div className="flex min-w-0 items-start justify-between gap-3">
                  <div className="min-w-0">
                    <CardTitle className="truncate text-base" title={artifact.title}>{artifact.title}</CardTitle>
                    <p className="mt-1 truncate font-mono text-xs text-text-secondary" title={artifact.sessionId}>
                      {artifact.sessionId}
                    </p>
                  </div>
                  <Badge tone="outline" className="shrink-0">{artifact.kind.toUpperCase()}</Badge>
                </div>
              </CardHeader>
              <CardContent className="flex min-w-0 flex-col gap-3 pt-0">
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-secondary">
                  <span>{artifact.language || "plain text"}</span>
                  <span>{formatBytes(artifactBytes(artifact))}</span>
                  <span>{formatArtifactDate(artifact.createdAt)}</span>
                </div>
                <div className="flex flex-wrap items-center gap-2 border-t border-border pt-3">
                  <Button type="button" size="sm" outlined onClick={() => setPreview(artifact)}>
                    <Eye /> Preview
                  </Button>
                  <Button type="button" size="sm" outlined onClick={() => void handleCopy(artifact)}>
                    {copiedId === artifact.id ? <Check /> : <Clipboard />}
                    {copiedId === artifact.id ? "Copied" : "Copy"}
                  </Button>
                  <Button type="button" size="sm" outlined onClick={() => downloadArtifact(artifact)}>
                    <Download /> Download
                  </Button>
                  <Button type="button" size="sm" ghost className="ml-auto" onClick={() => handleUnpin(artifact)}>
                    <Trash2 /> Unpin
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
