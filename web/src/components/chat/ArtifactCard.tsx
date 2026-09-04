import { useCallback, useEffect, useState } from "react";
import { Download, Eye, X } from "lucide-react";

import { useProfileScope } from "@/contexts/useProfileScope";
import { artifactDownloadName, type ArtifactDetection } from "@/lib/artifact-detect";
import { sandboxedArtifactDocument } from "@/lib/artifact-preview";
import {
  ARTIFACT_STORAGE_CHANGE_EVENT,
  getArtifactStorage,
  getArtifactStorageKey,
  isArtifactPinned,
  makeArtifactId,
  setArtifactPinnedWithResult,
  type ArtifactStorageFailureReason,
  type StoredArtifact,
} from "@/lib/artifact-storage";

export type ArtifactCardProps = {
  code: string;
  detection: ArtifactDetection;
  sessionId?: string;
  streaming?: boolean;
};

function artifactMime(kind: ArtifactDetection["kind"]): string {
  if (kind === "html") return "text/html;charset=utf-8";
  if (kind === "svg") return "image/svg+xml;charset=utf-8";
  return "text/plain;charset=utf-8";
}

function pinFailureMessage(reason: ArtifactStorageFailureReason): string {
  switch (reason) {
    case "artifact-too-large":
      return "This artifact is larger than the 200 KB per-artifact limit";
    case "storage-limit":
      return "Pinned artifact storage is full; remove an artifact first";
    case "storage-quota":
      return "The browser declined this storage change";
    case "storage-unavailable":
      return "Browser storage is unavailable";
  }
}

export function ArtifactCard({ code, detection, sessionId, streaming = false }: ArtifactCardProps) {
  const { profile, currentProfile } = useProfileScope();
  const artifactProfile = profile || currentProfile || "default";
  const [previewOpen, setPreviewOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [pinError, setPinError] = useState<string | null>(null);
  const [createdAt] = useState(() => Date.now());
  const filename = artifactDownloadName(detection.kind, detection.language, detection.title);
  const artifactId = makeArtifactId(sessionId ?? "unscoped", detection.kind, detection.language, detection.title, code);
  const storedArtifact: StoredArtifact = {
    id: artifactId,
    sessionId: sessionId ?? "unscoped",
    kind: detection.kind,
    language: detection.language,
    title: detection.title,
    code,
    createdAt,
  };
  const canPreview = detection.kind === "html" || detection.kind === "svg";
  const lineCount = code.trim().split("\n").length;

  const syncPinned = useCallback(() => {
    setPinned(Boolean(sessionId) && isArtifactPinned(getArtifactStorage(), artifactId, artifactProfile));
  }, [artifactId, artifactProfile, sessionId]);

  useEffect(() => {
    syncPinned();
    setPinError(null);
  }, [syncPinned]);

  useEffect(() => {
    const onArtifactStorageChange = (event: Event) => {
      const key = event.type === "storage"
        ? (event as StorageEvent).key ?? undefined
        : (event as CustomEvent<{ key?: string }>).detail?.key;
      if (key && key !== getArtifactStorageKey(artifactProfile)) return;
      syncPinned();
    };
    window.addEventListener(ARTIFACT_STORAGE_CHANGE_EVENT, onArtifactStorageChange);
    window.addEventListener("storage", onArtifactStorageChange);
    return () => {
      window.removeEventListener(ARTIFACT_STORAGE_CHANGE_EVENT, onArtifactStorageChange);
      window.removeEventListener("storage", onArtifactStorageChange);
    };
  }, [artifactProfile, syncPinned]);

  const togglePin = () => {
    if (!sessionId || streaming) return;
    const nextPinned = !pinned;
    const result = setArtifactPinnedWithResult(getArtifactStorage(), storedArtifact, nextPinned, artifactProfile);
    if (result.ok) {
      setPinned(nextPinned);
      setPinError(null);
    } else {
      setPinError(pinFailureMessage(result.reason));
    }
  };

  const download = () => {
    const blob = new Blob([code], { type: artifactMime(detection.kind) });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  return (
    <section
      data-slot="artifact-card"
      data-artifact-kind={detection.kind}
      data-artifact-pinned={pinned ? "true" : "false"}
      aria-label={`${detection.kind} artifact: ${detection.title}`}
      className="my-2 overflow-hidden rounded-md border border-border bg-card text-card-foreground"
    >
      <div className="flex min-w-0 items-center gap-3 px-3 py-2">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded bg-muted text-muted-foreground" aria-hidden>
          {detection.kind === "html" ? "◇" : detection.kind === "svg" ? "△" : "{}"}
        </div>
        <div className="min-w-0 flex-1">
          <strong className="block truncate text-sm">{detection.title}</strong>
          <span className="block truncate text-xs text-muted-foreground">
            {detection.kind.toUpperCase()} · {lineCount} lines{streaming ? " · generating" : ""}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            aria-label={pinned ? "Unpin artifact" : "Pin artifact"}
            className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!sessionId || streaming}
            onClick={togglePin}
          >
            {pinned ? "Pinned" : "Pin"}
          </button>
          {canPreview && (
            <button
              type="button"
              aria-label="Preview artifact"
              className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-50"
              disabled={streaming}
              onClick={() => setPreviewOpen(true)}
            >
              <Eye aria-hidden />
              Preview
            </button>
          )}
          <button
            type="button"
            aria-label="Download artifact"
            className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-50"
            disabled={streaming}
            onClick={download}
          >
            <Download aria-hidden />
            Download
          </button>
        </div>
      </div>
      {pinError && <div role="alert" className="border-t border-border px-3 py-1 text-xs text-warning">{pinError}</div>}
      {previewOpen && canPreview && (
        <div data-slot="artifact-preview" role="dialog" aria-label={`${detection.title} preview`} className="border-t border-border bg-background p-2">
          <div className="mb-2 flex items-center justify-between gap-2 px-1">
            <span className="text-xs text-muted-foreground">Sandboxed preview</span>
            <button type="button" aria-label="Close artifact preview" className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-accent-foreground" onClick={() => setPreviewOpen(false)}>
              <X aria-hidden />
            </button>
          </div>
          <iframe
            title={`${detection.title} preview`}
            sandbox=""
            referrerPolicy="no-referrer"
            srcDoc={sandboxedArtifactDocument(code)}
            className="h-80 w-full rounded border border-border bg-white"
          />
        </div>
      )}
    </section>
  );
}
