import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { useI18n } from "@/i18n";
import { AlertTriangle } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cn, themedBody } from "@/lib/utils";

interface ConfirmDialogProps {
  cancelLabel?: string;
  confirmLabel?: string;
  description?: string;
  destructive?: boolean;
  loading?: boolean;
  onCancel: () => void;
  onConfirm: () => Promise<void> | void;
  open: boolean;
  title: string;
  typedConfirmation?: string;
}

export function ConfirmDialog(props: ConfirmDialogProps) {
  return props.open ? <OpenConfirmDialog key={props.typedConfirmation} {...props} /> : null;
}

function OpenConfirmDialog({
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
  const { t } = useI18n();
  const submittingRef = useRef(false);
  const inputId = useId();
  const [confirmation, setConfirmation] = useState("");
  const ready = typedConfirmation === undefined || confirmation === typedConfirmation;
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const prevActive = document.activeElement as HTMLElement | null;
    dialogRef.current
      ?.querySelector<HTMLElement>(typedConfirmation !== undefined ? "input" : "[data-confirm]")
      ?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !loading) {
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
  }, [open, onCancel, typedConfirmation, loading]);

  const run = async () => {
    if (!ready || loading || submittingRef.current) return;
    submittingRef.current = true;
    try {
      await onConfirm();
    } finally {
      submittingRef.current = false;
    }
  };

  if (!open) return null;

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
      aria-describedby={description ? "confirm-dialog-desc" : undefined}
      onClick={(e) => {
        if (e.target === e.currentTarget && !loading) onCancel();
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

        {typedConfirmation !== undefined && (
          <div className="flex flex-col gap-2 px-4 pt-4">
            <label className="text-xs text-muted-foreground" htmlFor={inputId}>
              {t.common.typedConfirmation(typedConfirmation)}
            </label>
            <Input
              autoComplete="off"
              disabled={loading}
              id={inputId}
              onChange={(event) => setConfirmation(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  void run();
                }
              }}
              spellCheck={false}
              value={confirmation}
            />
          </div>
        )}
        <div className="flex items-center justify-end gap-2 p-3">
          <Button type="button" outlined onClick={onCancel} disabled={loading}>
            {cancelLabel}
          </Button>
          <Button
            data-confirm
            type="button"
            destructive={destructive}
            onClick={() => void run()}
            disabled={loading || !ready}
          >
            {loading ? "…" : confirmLabel}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
