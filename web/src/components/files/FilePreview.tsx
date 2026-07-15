import { FileIcon } from "lucide-react";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { FileMarkdown } from "@/components/FileMarkdown";
import type { ManagedFileEntry, ManagedFileReadResponse } from "@/lib/api";
import type { PreviewKind } from "@/lib/file-browser";

interface FilePreviewProps {
  entry: ManagedFileEntry;
  kind: PreviewKind;
  file: ManagedFileReadResponse | null;
  text: string;
  mode: "rendered" | "source";
  loading: boolean;
  error: string | null;
}

export function FilePreview({
  entry,
  kind,
  file,
  text,
  mode,
  loading,
  error,
}: FilePreviewProps) {
  if (loading) {
    return (
      <div className="flex min-h-80 items-center justify-center gap-2 text-sm text-text-secondary">
        <Spinner /> Loading preview...
      </div>
    );
  }
  if (error) {
    return <div className="p-4 text-sm text-destructive">{error}</div>;
  }
  if (kind === "large") {
    return (
      <div className="flex min-h-80 flex-col items-center justify-center gap-2 p-6 text-center">
        <FileIcon className="h-9 w-9 text-text-tertiary" />
        <p className="font-medium">File too large to preview</p>
        <p className="text-sm text-text-secondary">Download file to inspect its contents.</p>
      </div>
    );
  }
  if (kind === "binary") {
    return (
      <div className="flex min-h-80 flex-col items-center justify-center gap-2 p-6 text-center">
        <FileIcon className="h-9 w-9 text-text-tertiary" />
        <p className="font-medium">Preview unavailable</p>
        <p className="text-sm text-text-secondary">{entry.mime_type || "Binary file"}</p>
      </div>
    );
  }
  if (kind === "image" && file) {
    return (
      <div className="flex min-h-80 items-center justify-center bg-background/30 p-6">
        <img
          src={file.data_url}
          alt={entry.name}
          className="max-h-[65dvh] max-w-full border border-border object-contain"
        />
      </div>
    );
  }
  if (kind === "markdown" && mode === "rendered") {
    return (
      <div className="overflow-x-auto p-5 sm:p-8">
        <FileMarkdown content={text} />
      </div>
    );
  }
  return (
    <pre className="min-h-80 overflow-auto bg-background/25 p-4 font-mono text-xs leading-6 text-foreground">
      <code>{text}</code>
    </pre>
  );
}
