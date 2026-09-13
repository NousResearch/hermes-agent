/**
 * ReasoningPickerDialog — overlay modal for choosing reasoning effort.
 *
 * Mirrors ModelPickerDialog's portal-to-body overlay pattern: the dashboard's
 * chat sidebar column is `relative z-2`, which traps fixed-position
 * descendants below the app sidebar (z-50) and clips a plain dropdown's
 * popover to the card's small div. Portaling to document.body with a
 * fixed inset-0 overlay sidesteps both problems, same as the model picker.
 *
 * Accessibility: on open, focus moves to the first option (falls back to the
 * close button if the list is somehow empty) and Tab/Shift+Tab cycle within
 * the dialog's focusable elements so keyboard users can't tab back into the
 * sidebar content hidden behind the overlay. On close, focus returns to
 * whatever had it before the dialog opened (the trigger button in practice),
 * matching standard modal behavior.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { X } from "lucide-react";
import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";

import { cn, themedBody } from "@/lib/utils";
import { EFFORT_OPTIONS } from "@/lib/reasoning-effort";

interface Props {
  currentEffort: string;
  /** May return a promise (the config save) — the dialog waits for it to
   *  settle before closing, so a failed save's revert still happens with
   *  the dialog visible rather than already dismissed. */
  onSelect(effort: string): void | Promise<void>;
  onClose(): void;
}

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

export function ReasoningPickerDialog({
  currentEffort,
  onSelect,
  onClose,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const firstOptionRef = useRef<HTMLButtonElement | null>(null);
  // Element that had focus before the dialog opened (the trigger button),
  // restored on close.
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  // Focus the first option on mount (falls back to any focusable element in
  // the dialog, e.g. the close button, if the option list were ever empty);
  // restore focus to the trigger on unmount.
  useEffect(() => {
    previouslyFocusedRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const fallback = containerRef.current?.querySelector<HTMLElement>(
      FOCUSABLE_SELECTOR,
    );
    (firstOptionRef.current ?? fallback)?.focus();
    return () => {
      previouslyFocusedRef.current?.focus();
    };
  }, []);

  // Esc closes; Tab/Shift+Tab cycle focus within the dialog instead of
  // escaping into the sidebar content behind the overlay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab" || !containerRef.current) return;
      const focusable = Array.from(
        containerRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const currentLabel =
    EFFORT_OPTIONS.find((o) => o.value === currentEffort)?.label ??
    currentEffort;

  const selectAndClose = (value: string) => {
    // Await the save (if onSelect returns a promise) before dismissing, so
    // a failed save's revert-on-failure path still runs while the dialog
    // (and its "saving" affordance upstream) is visible.
    void Promise.resolve(onSelect(value)).finally(() => onClose());
  };

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="reasoning-picker-title"
    >
      <div
        ref={containerRef}
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
            current: {currentLabel}
          </p>
        </header>

        <div className="flex-1 min-h-0 overflow-y-auto p-2">
          {EFFORT_OPTIONS.map((opt, index) => (
            <ListItem
              key={opt.value}
              ref={index === 0 ? firstOptionRef : undefined}
              active={opt.value === currentEffort}
              onClick={() => selectAndClose(opt.value)}
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
