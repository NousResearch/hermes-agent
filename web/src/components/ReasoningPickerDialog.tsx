/**
 * ReasoningPickerDialog — overlay modal for choosing reasoning effort.
 *
 * Mirrors ModelPickerDialog's portal-to-body overlay pattern: the dashboard's
 * chat sidebar column is `relative z-2`, which traps fixed-position
 * descendants below the app sidebar (z-50) and clips a plain dropdown's
 * popover to the card's small div. Portaling to document.body with a
 * fixed inset-0 overlay sidesteps both problems, same as the model picker.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { X } from "lucide-react";
import { useEffect } from "react";
import { createPortal } from "react-dom";

import { cn, themedBody } from "@/lib/utils";
import { EFFORT_OPTIONS } from "@/lib/reasoning-effort";

interface Props {
  currentEffort: string;
  onSelect(effort: string): void;
  onClose(): void;
}

export function ReasoningPickerDialog({
  currentEffort,
  onSelect,
  onClose,
}: Props) {
  // Esc closes — same UX as ModelPickerDialog.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="reasoning-picker-title"
    >
      <div
        className={cn(
          themedBody,
          "relative w-full max-w-sm max-h-[80vh] border border-border bg-card shadow-2xl flex flex-col",
        )}
      >
        <Button
          ghost
          size="icon"
          onClick={onClose}
          className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
          aria-label="Close"
        >
          <X />
        </Button>

        <header className="p-5 pb-3 border-b border-border">
          <h2
            id="reasoning-picker-title"
            className="font-mondwest text-display text-base tracking-wider"
          >
            Reasoning Effort
          </h2>
          <p className="text-xs text-muted-foreground mt-1 font-mono">
            current: {currentEffort}
          </p>
        </header>

        <div className="flex-1 min-h-0 overflow-y-auto p-2">
          {EFFORT_OPTIONS.map((opt) => (
            <ListItem
              key={opt.value}
              active={opt.value === currentEffort}
              onClick={() => {
                onSelect(opt.value);
                onClose();
              }}
              className="cursor-pointer"
            >
              {opt.label}
            </ListItem>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  );
}
