import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export function ConfirmDialog({
  cancelLabel = "Cancel",
  confirmLabel = "Confirm",
  description,
  destructive = false,
  loading = false,
  onCancel,
  onConfirm,
  open,
  title,
}: ConfirmDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const prevActive = document.activeElement as HTMLElement | null;
    dialogRef.current
      ?.querySelector<HTMLButtonElement>("[data-confirm]")
      ?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
    };

    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      prevActive?.focus?.();
    };
  }, [open, onCancel]);

  if (!open) return null;

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      aria-describedby={description ? "confirm-dialog-desc" : undefined}
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
      className={cn(
        "fixed inset-0 z-50 flex items-center justify-center",
        "bg-black/60 backdrop-blur-sm",
        "animate-[fade-in_150ms_ease-out]",
      )}
    >
      <div
        ref={dialogRef}
        className={cn(
          "relative mx-4 w-full max-w-md rounded-xl border border-border bg-card text-card-foreground shadow-lg",
          "animate-[dialog-in_180ms_ease-out]",
        )}
      >
        <div className="flex items-start gap-3 border-b border-border p-5">
          {destructive && (
            <div
              aria-hidden
              className="mt-0.5 shrink-0 rounded-full bg-destructive/10 p-2 text-destructive"
            >
              <AlertTriangle className="h-4 w-4" />
            </div>
          )}

          <div className="flex min-w-0 flex-1 flex-col gap-2">
            <h2
              id="confirm-dialog-title"
              className="text-base font-semibold tracking-tight text-card-foreground"
            >
              {title}
            </h2>

            {description && (
              <p
                id="confirm-dialog-desc"
                className="text-sm leading-relaxed text-muted-foreground"
              >
                {description}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 p-4">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onCancel}
            disabled={loading}
          >
            {cancelLabel}
          </Button>
          <Button
            data-confirm
            type="button"
            variant={destructive ? "destructive" : "default"}
            size="sm"
            onClick={onConfirm}
            disabled={loading}
          >
            {loading ? "…" : confirmLabel}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

interface ConfirmDialogProps {
  cancelLabel?: string;
  confirmLabel?: string;
  description?: string;
  destructive?: boolean;
  loading?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  open: boolean;
  title: string;
}
