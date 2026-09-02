import { Button } from "@nous-research/ui/ui/components/button";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface EditMessageDialogProps {
  initialText: string;
  loading: boolean;
  onCancel: () => void;
  onConfirm: (text: string) => void;
  open: boolean;
}

export function EditMessageDialog({ initialText, loading, onCancel, onConfirm, open }: EditMessageDialogProps) {
  const [text, setText] = useState(initialText);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!open) return;
    textareaRef.current?.focus();
    textareaRef.current?.setSelectionRange(textareaRef.current.value.length, textareaRef.current.value.length);
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCancel();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [onCancel, open]);

  if (!open) return null;
  const canConfirm = text.trim().length > 0 && !loading;
  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="edit-message-dialog-title"
      className="fixed inset-0 z-[200] flex items-center justify-center bg-background/85 p-4"
      onClick={(event) => {
        if (event.target === event.currentTarget && !loading) onCancel();
      }}
    >
      <div className="w-full max-w-lg border border-border bg-card shadow-2xl">
        <div className="border-b border-border p-4">
          <h2 id="edit-message-dialog-title" className="font-mondwest text-display text-base tracking-wider">
            Edit and regenerate this turn?
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            This replaces the selected user turn and all later turns in this session. The gateway will keep the dropped rows archived.
          </p>
        </div>
        <div className="p-4">
          <label htmlFor="edit-message-text" className="sr-only">Edited user message</label>
          <textarea
            ref={textareaRef}
            id="edit-message-text"
            aria-label="Edited user message"
            className="min-h-28 w-full resize-y rounded border border-border bg-background p-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            value={text}
            onChange={(event) => setText(event.target.value)}
            disabled={loading}
          />
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-border p-3">
          <Button type="button" outlined onClick={onCancel} disabled={loading}>Cancel</Button>
          <Button type="button" data-confirm destructive onClick={() => onConfirm(text.trim())} disabled={!canConfirm}>
            {loading ? "…" : "Edit and regenerate"}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
