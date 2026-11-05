import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { SendHorizonal } from "lucide-react";

import { api } from "@/lib/api";
import {
  imageFilesFromTransfer,
  transferMayContainImage,
  uploadChatImage,
} from "@/lib/chatImagePaste";
import { cn } from "@/lib/utils";

interface PendingAttachment {
  path: string;
  name: string;
}

interface ChatBarProps {
  onSend: (text: string, attachments: string[]) => void;
  disabled?: boolean;
  profile?: string;
}

export function ChatBar({ onSend, disabled, profile }: ChatBarProps) {
  const [text, setText] = useState("");
  const [model, setModel] = useState("");
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [uploading, setUploading] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const uploadSeq = useRef(0);

  useEffect(() => {
    let cancelled = false;
    api
      .getModelInfo(profile)
      .then((r) => {
        if (!cancelled && r?.model) setModel(String(r.model));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [profile]);

  const send = useCallback(() => {
    const value = text.trim();
    if ((!value && attachments.length === 0) || disabled || uploading > 0)
      return;
    const paths = attachments.map((a) => a.path);
    setText("");
    setAttachments([]);
    onSend(value, paths);
  }, [text, attachments, disabled, uploading, onSend]);

  const addFiles = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;
      setUploadError(null);
      for (const file of files) {
        const seq = ++uploadSeq.current;
        setUploading((n) => n + 1);
        uploadChatImage(file, profile ?? "")
          .then((res) => {
            setAttachments((prev) => [
              ...prev,
              { path: res.path, name: res.name },
            ]);
          })
          .catch((e: Error) => {
            setUploadError(e.message || `upload ${seq} failed`);
          })
          .finally(() => {
            setUploading((n) => Math.max(0, n - 1));
          });
      }
    },
    [profile],
  );

  const removeAttachment = useCallback((path: string) => {
    setAttachments((prev) => prev.filter((a) => a.path !== path));
  }, []);

  return (
    <div
      className="flex shrink-0 flex-col gap-2 border-t border-current/10 pt-2"
      onDragOver={(e) => {
        if (transferMayContainImage(e.dataTransfer)) e.preventDefault();
      }}
      onDrop={(e) => {
        const files = imageFilesFromTransfer(e.dataTransfer);
        if (files.length > 0) {
          e.preventDefault();
          addFiles(files);
        }
      }}
    >
      {model && (
        <span className="w-fit rounded-full border border-current/20 px-2 py-0.5 text-[0.6875rem] text-text-secondary">
          {model}
        </span>
      )}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {attachments.map((a) => (
            <span
              key={a.path}
              className="inline-flex max-w-48 items-center gap-1 truncate rounded-full border border-current/20 px-2 py-0.5 text-[0.6875rem] text-text-secondary"
            >
              <span className="truncate">{a.name}</span>
              <button
                type="button"
                onClick={() => removeAttachment(a.path)}
                aria-label={`Remove ${a.name}`}
                className="hover:text-midground"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      {uploading > 0 && (
        <span className="text-[0.6875rem] text-text-secondary">
          Uploading image…
        </span>
      )}
      {uploadError && (
        <span className="text-[0.6875rem] text-destructive">{uploadError}</span>
      )}
      <div className="flex items-end gap-2">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onPaste={(e) => addFiles(imageFilesFromTransfer(e.clipboardData))}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          disabled={disabled}
          placeholder="Message… (paste or drop images)"
          aria-label="Message"
          className={cn(
            "min-h-10 max-h-40 flex-1 resize-y rounded border border-current/20",
            "bg-background-base px-3 py-2 text-sm text-foreground",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
          )}
        />
        <Button
          outlined
          size="sm"
          onClick={send}
          disabled={
            disabled || uploading > 0 || (!text.trim() && attachments.length === 0)
          }
          prefix={<SendHorizonal />}
          aria-label="Send"
        >
          Send
        </Button>
      </div>
    </div>
  );
}
