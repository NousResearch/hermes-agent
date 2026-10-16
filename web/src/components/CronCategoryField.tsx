import { useMemo, useRef, useState } from "react";

import { Input } from "@nous-research/ui/ui/components/input";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";

interface CronCategoryFieldProps {
  id: string;
  /** Current value — a single label, "" when unset. */
  value: string;
  /** Labels already in use, for the autocomplete. */
  options: string[];
  onChange: (value: string) => void;
}

/** Tag-style category picker: type to filter the labels already in use, pick one
 *  to reuse it, or keep typing and confirm a new one. */
export function CronCategoryField({ id, value, options, onChange }: CronCategoryFieldProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const wrapRef = useRef<HTMLDivElement>(null);

  const typed = value.trim();
  const matches = useMemo(
    () => options.filter((option) => option.toLowerCase().includes(typed.toLowerCase())),
    [options, typed],
  );
  // Offer the new label only when it is not already one of the existing ones
  // (case-insensitively — otherwise picking it would just create a twin).
  const canCreate =
    typed.length > 0 && !options.some((option) => option.toLowerCase() === typed.toLowerCase());
  const rows = [
    ...matches.map((label) => ({ kind: "option" as const, label })),
    ...(canCreate ? [{ kind: "create" as const, label: typed }] : []),
  ];
  const listVisible = open && rows.length > 0;

  const pick = (label: string) => {
    onChange(label);
    setOpen(false);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Escape") {
      setOpen(false);
      return;
    }
    if (!listVisible) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlight((index) => (index + 1) % rows.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlight((index) => (index - 1 + rows.length) % rows.length);
    } else if (event.key === "Enter") {
      const row = rows[Math.min(highlight, rows.length - 1)];
      if (row) {
        event.preventDefault();
        pick(row.label);
      }
    }
  };

  return (
    <div
      ref={wrapRef}
      className="relative"
      // Close when focus leaves the field entirely (clicking a row keeps focus inside).
      onBlur={(event) => {
        if (!wrapRef.current?.contains(event.relatedTarget as Node | null)) setOpen(false);
      }}
    >
      <Input
        id={id}
        value={value}
        autoComplete="off"
        placeholder={t.cron.categoryPlaceholder ?? en.cron.categoryPlaceholder}
        role="combobox"
        aria-expanded={listVisible}
        aria-autocomplete="list"
        aria-controls={`${id}-list`}
        onChange={(event) => {
          onChange(event.target.value);
          setOpen(true);
          setHighlight(0);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
      />
      {value.length > 0 && (
        <button
          type="button"
          aria-label={t.cron.categoryClear ?? en.cron.categoryClear}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          onClick={() => {
            onChange("");
            setOpen(false);
          }}
        >
          ×
        </button>
      )}
      {listVisible && (
        <div
          id={`${id}-list`}
          role="listbox"
          aria-label={t.cron.categoryExisting ?? en.cron.categoryExisting}
          className="absolute z-50 mt-1 max-h-56 w-full overflow-auto border border-border bg-background shadow-lg"
        >
          {rows.map((row, index) => {
            const isCurrent = row.label === value.trim();
            return (
              <button
                key={`${row.kind}:${row.label}`}
                type="button"
                role="option"
                // aria-selected marks the job's actual category (this is a
                // single-select list); `data-highlighted` is only the cursor row.
                aria-selected={isCurrent}
                data-highlighted={index === highlight ? "true" : undefined}
                data-testid={`cron-category-${row.kind}`}
                className={cn(
                  "flex w-full items-center gap-2 px-3 py-2 text-left text-sm",
                  index === highlight ? "bg-muted" : "bg-transparent",
                )}
                onMouseEnter={() => setHighlight(index)}
                onClick={() => pick(row.label)}
              >
                {row.kind === "create" && (
                  <span className="text-xs text-muted-foreground">
                    {t.cron.categoryCreateNew ?? en.cron.categoryCreateNew}
                  </span>
                )}
                <span className="truncate">{row.label}</span>
                {isCurrent && (
                  <span className="ml-auto text-xs text-muted-foreground">
                    {t.cron.categoryCurrent ?? en.cron.categoryCurrent}
                  </span>
                )}
              </button>
            );
          })}
          {value.trim().length > 0 && (
            <p className="border-t border-border px-3 py-2 text-xs text-muted-foreground">
              {t.cron.categorySingleHint ?? en.cron.categorySingleHint}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
