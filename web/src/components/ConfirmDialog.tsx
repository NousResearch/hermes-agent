import { Button } from "@nous-research/ui/ui/components/button";
import { AlertTriangle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
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
  typedConfirmation?: string;
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
  typedConfirmation,
}: ConfirmDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const [typedValue, setTypedValue] = useState("");
  const cancel = useCallback(() => {
    setTypedValue("");
    onCancel();
  }, [onCancel]);
  const confirm = () => {
    setTypedValue("");
    onConfirm();
  };

  useEffect(() => {
    if (!open) return;

    const prevActive = document.activeElement as HTMLElement | null;
    dialogRef.current
      ?.querySelector<HTMLButtonElement>("[data-confirm]")
      ?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        cancel();
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
  }, [open, cancel]);

  if (!open) return null;

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      aria-describedby={description ? "confirm-dialog-desc" : undefined}
      onClick={(e) => {
        if (e.target === e.currentTarget) cancel();
      }}
      className="fixed inset-0 z-[200] flex items-center justify-center bg-background/85 p-4"
    >
      <div
        ref={dialogRef}
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

        {typedConfirmation && (
          <label className="flex flex-col gap-2 px-4 pt-4 text-xs text-muted-foreground">
            Type{" "}
            <strong className="text-foreground">{typedConfirmation}</strong> to
            confirm
            <input
              autoComplete="off"
              className="border border-border bg-background px-3 py-2 text-foreground"
              onChange={(event) => setTypedValue(event.target.value)}
              value={typedValue}
            />
          </label>
        )}

        <div className="flex items-center justify-end gap-2 p-3">
          <Button type="button" outlined onClick={cancel} disabled={loading}>
            {cancelLabel}
          </Button>
          <Button
            data-confirm
            type="button"
            destructive={destructive}
            onClick={confirm}
            disabled={
              loading ||
              Boolean(
                typedConfirmation && typedValue !== typedConfirmation,
              )
            }
          >
            {loading ? "…" : confirmLabel}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
