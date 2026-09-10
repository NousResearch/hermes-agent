import { useRef, type ReactNode } from "react";
import { Upload, X } from "lucide-react";
import { ChatToolbarButton } from "./ChatToolbarButton";
import type { useChatFileUploads } from "@/hooks/useChatFileUploads";

interface ChatFileUploadsProps extends ReturnType<typeof useChatFileUploads> {
  terminalForeground: string;
  copyAction: ReactNode;
}

export function ChatFileUploads({ uploads, select, retry, remove, available, terminalForeground, copyAction }: ChatFileUploadsProps) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <section aria-label="File uploads" className="shrink-0 pt-1 text-xs">
      <div className="max-h-24 overflow-y-auto text-text-secondary" aria-live="polite" aria-relevant="additions text">
        {uploads.map(entry => (
          <div key={entry.id} className="mb-1 flex items-center gap-2" role="status">
            <span className="min-w-0 truncate">{entry.name}</span>
            <span className={entry.error ? "text-destructive" : undefined}>
              {entry.error ?? ({ queued: "Waiting…", uploading: "Uploading…", attaching: "Attaching…" } as Partial<Record<typeof entry.state, string>>)[entry.state]}
            </span>
            {entry.canRetry && <button type="button" className="shrink-0 px-1 py-1 underline hover:text-text-primary" onClick={() => retry(entry.id)}>Retry</button>}
            {entry.state !== "attaching" && (
              <button type="button" className="shrink-0 p-1 hover:text-text-primary" aria-label={`Dismiss upload ${entry.name}`} title={entry.state === "unknown" ? "Dismiss status only; the file may already be attached" : undefined} onClick={() => remove(entry.id)}>
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        ))}
      </div>
      <input
        ref={input} type="file" multiple hidden disabled={!available}
        aria-label="Choose files to upload"
        onChange={event => {
          const files = Array.from(event.currentTarget.files ?? []);
          event.currentTarget.value = "";
          select(files);
        }}
      />
      <div className="flex items-center justify-between gap-2">
        <ChatToolbarButton terminalForeground={terminalForeground} icon={Upload} aria-label="Upload files" disabled={!available} onClick={() => input.current?.click()}>
          Upload
        </ChatToolbarButton>
        {copyAction}
      </div>
    </section>
  );
}
