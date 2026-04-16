import { useState, useRef, useEffect, useCallback } from "react";
import { Palette, Check } from "lucide-react";
import { useTheme } from "@/themes";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * Compact theme picker for the dashboard header.
 * Shows a palette icon + current theme name; opens a dropdown of all
 * available themes with color swatches for instant preview.
 */
export function ThemeSwitcher() {
  const { themeName, availableThemes, setTheme } = useTheme();
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const close = useCallback(() => setOpen(false), []);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, close]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, close]);

  const current = availableThemes.find((t) => t.name === themeName);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "group relative inline-flex items-center gap-1.5 px-2 py-1 text-xs",
          "text-muted-foreground hover:text-foreground transition-colors",
          "cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        )}
        title={t.theme?.switchTheme ?? "Switch theme"}
        aria-label={t.theme?.switchTheme ?? "Switch theme"}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <Palette className="h-3.5 w-3.5" />
        <span className="hidden sm:inline font-display tracking-wide uppercase text-[0.65rem]">
          {current?.label ?? themeName}
        </span>
      </button>

      {open && (
        <div
          role="listbox"
          className={cn(
            "absolute right-0 top-full mt-1 z-50 min-w-[200px]",
            "border border-border bg-popover text-popover-foreground shadow-lg",
            "animate-[fade-in_100ms_ease-out]",
          )}
        >
          <div className="px-3 py-2 border-b border-border">
            <span className="font-display text-[0.7rem] tracking-[0.12em] uppercase text-muted-foreground">
              {t.theme?.title ?? "Theme"}
            </span>
          </div>

          {availableThemes.map((theme) => {
            const isActive = theme.name === themeName;
            return (
              <button
                key={theme.name}
                type="button"
                role="option"
                aria-selected={isActive}
                onClick={() => {
                  setTheme(theme.name);
                  close();
                }}
                className={cn(
                  "flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm transition-colors cursor-pointer",
                  "hover:bg-foreground/10",
                  isActive ? "text-foreground" : "text-muted-foreground",
                )}
              >
                <Check
                  className={cn(
                    "h-3 w-3 shrink-0",
                    isActive ? "opacity-100" : "opacity-0",
                  )}
                />
                <div className="flex flex-col gap-0.5 min-w-0">
                  <span className="font-medium text-xs truncate">{theme.label}</span>
                  <span className="text-[0.65rem] text-muted-foreground truncate">
                    {theme.description}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
