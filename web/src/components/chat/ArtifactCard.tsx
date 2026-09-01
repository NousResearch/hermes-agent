import { useState } from "react";
import { Download, Eye, X } from "lucide-react";

import { artifactDownloadName, type ArtifactDetection } from "@/lib/artifact-detect";

export type ArtifactCardProps = {
  code: string;
  detection: ArtifactDetection;
  streaming?: boolean;
};

function artifactMime(kind: ArtifactDetection["kind"]): string {
  if (kind === "html") return "text/html;charset=utf-8";
  if (kind === "svg") return "image/svg+xml;charset=utf-8";
  return "text/plain;charset=utf-8";
}

export function ArtifactCard({ code, detection, streaming = false }: ArtifactCardProps) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const filename = artifactDownloadName(detection.kind, detection.language, detection.title);
  const canPreview = detection.kind === "html" || detection.kind === "svg";
  const lineCount = code.trim().split("\n").length;

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
            sandbox="allow-scripts"
            srcDoc={code}
            className="h-80 w-full rounded border border-border bg-white"
          />
        </div>
      )}
    </section>
  );
}
