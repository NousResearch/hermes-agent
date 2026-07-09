import { Button } from "@nous-research/ui/ui/components/button";
import { AlertTriangle } from "lucide-react";
import { createPortal } from "react-dom";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { cn, themedBody } from "@/lib/utils";

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
  const dialogRef = useModalBehavior({ open, onClose: onCancel });

  if (!open) return null;

  return createPortal(
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      aria-describedby={description ? "confirm-dialog-desc" : undefined}
      tabIndex={-1}
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
      className="fixed inset-0 z-[200] flex items-center justify-center bg-background/85 p-4"
    >
      <div
        className={cn(
          themedBody,
          "relative w-full max-w-md border border-border bg-card shadow-2xl",
        )}
      >
        <div className="flex items-start gap-3 p-4 border-b border-border">
          {destructive && (
            <div aria-hidden className="mt-0.5 shrink-0 text-destructive">
              <AlertTriangle className="h-4 w-4" />
            </div>
          )}

          <div className="flex-1 min-w-0 flex flex-col gap-1">
            <h2
              id="confirm-dialog-title"
              className="font-mondwest text-display text-base tracking-wider"
            >
              {title}
            </h2>

            {description && (
              <p
                id="confirm-dialog-desc"
                className="text-xs text-muted-foreground leading-relaxed whitespace-pre-line"
              >
                {description}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 p-3">
          <Button type="button" outlined onClick={onCancel} disabled={loading}>
            {cancelLabel}
          </Button>
          <Button
            data-autofocus
            data-confirm
            type="button"
            destructive={destructive}
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
