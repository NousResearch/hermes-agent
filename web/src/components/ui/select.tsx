import { useState, useRef, useEffect, useCallback } from "react";
import { ChevronDown, Check } from "lucide-react";
import { cn } from "@/lib/utils";

export function Select({
  value,
  onValueChange,
  children,
  className,
  id,
  disabled,
}: SelectProps) {
  const [open, setOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const options: SelectOptionData[] = [];
  flattenChildren(children, options);

  const selectedOption = options.find((o) => o.value === value);
  const displayLabel = selectedOption?.label ?? value ?? "";

  const close = useCallback(() => {
    setOpen(false);
    setHighlightedIndex(-1);
  }, []);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        close();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, close]);

  useEffect(() => {
    if (open && listRef.current && highlightedIndex >= 0) {
      const el = listRef.current.children[highlightedIndex] as HTMLElement | undefined;
      el?.scrollIntoView({ block: "nearest" });
    }
  }, [open, highlightedIndex]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    switch (e.key) {
      case "Enter":
      case " ":
        e.preventDefault();
        if (!open) {
          setOpen(true);
          setHighlightedIndex(options.findIndex((o) => o.value === value));
        } else if (highlightedIndex >= 0 && options[highlightedIndex]) {
          onValueChange?.(options[highlightedIndex].value);
          close();
        }
        break;
      case "ArrowDown":
        e.preventDefault();
        if (!open) {
          setOpen(true);
          setHighlightedIndex(options.findIndex((o) => o.value === value));
        } else {
          setHighlightedIndex((i) => Math.min(i + 1, options.length - 1));
        }
        break;
      case "ArrowUp":
        e.preventDefault();
        if (open) {
          setHighlightedIndex((i) => Math.max(i - 1, 0));
        }
        break;
      case "Escape":
        e.preventDefault();
        close();
        break;
    }
  };

  return (
    <div ref={containerRef} className={cn("relative", className)} id={id}>
      <button
        type="button"
        role="combobox"
        aria-expanded={open}
        aria-haspopup="listbox"
        disabled={disabled}
        onClick={() => !disabled && setOpen((o) => !o)}
        onKeyDown={handleKeyDown}
        className={cn(
          "flex h-9 w-full items-center justify-between border border-border bg-background/40 px-3 py-1 font-courier text-sm text-left transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "cursor-pointer",
        )}
      >
        <span className={cn("truncate", !selectedOption && "text-muted-foreground")}>
          {displayLabel}
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <div
          ref={listRef}
          role="listbox"
          className={cn(
            "absolute z-50 mt-1 w-full border border-border bg-popover text-popover-foreground shadow-lg",
            "max-h-60 overflow-auto",
            "animate-[fade-in_100ms_ease-out]",
          )}
        >
          {options.map((opt, i) => {
            const isSelected = opt.value === value;
            const isHighlighted = i === highlightedIndex;
            return (
              <div
                key={opt.value}
                role="option"
                aria-selected={isSelected}
                onMouseEnter={() => setHighlightedIndex(i)}
                onClick={() => {
                  onValueChange?.(opt.value);
                  close();
                }}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 text-sm font-courier cursor-pointer transition-colors",
                  isHighlighted && "bg-foreground/10",
                  isSelected && "text-foreground",
                  !isSelected && "text-muted-foreground",
                )}
              >
                <Check
                  className={cn(
                    "h-3.5 w-3.5 shrink-0",
                    isSelected ? "opacity-100" : "opacity-0",
                  )}
                />
                <span className="truncate">{opt.label}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function SelectOption(_props: SelectOptionProps) {
  return null;
}

function flattenChildren(children: React.ReactNode, out: SelectOptionData[]) {
  const arr = Array.isArray(children) ? children : [children];
  for (const child of arr) {
    if (!child || typeof child !== "object" || !("props" in child)) continue;
    const props = child.props as Record<string, unknown>;
    if (props.value !== undefined) {
      out.push({
        value: String(props.value),
        label: typeof props.children === "string" ? props.children : String(props.value),
      });
    } else if (props.children) {
      flattenChildren(props.children as React.ReactNode, out);
    }
  }
}

interface SelectProps {
  value?: string;
  onValueChange?: (value: string) => void;
  children?: React.ReactNode;
  className?: string;
  id?: string;
  disabled?: boolean;
}

interface SelectOptionProps {
  value: string;
  children: React.ReactNode;
}

interface SelectOptionData {
  value: string;
  label: string;
}
