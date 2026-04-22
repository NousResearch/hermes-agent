import { useCallback, useEffect, useRef, useState } from "react";
import { Palette, Check } from "lucide-react";
import { BUILTIN_THEMES, useTheme } from "@/themes";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * Compact theme picker mounted next to the language switcher in the header.
 * Each dropdown row shows a 3-stop swatch (background / midground / warm
 * glow) so users can preview the palette before committing. User-defined
 * themes from `~/.hermes/dashboard-themes/*.yaml` that aren't in
 * `BUILTIN_THEMES` render without swatches and apply the default palette.
 */
export function ThemeSwitcher() {
  const { themeName, availableThemes, setTheme } = useTheme();
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(e.target as Node)
      ) {
        close();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, close]);

  const current = availableThemes.find((th) => th.name === themeName);
  const label = current?.label ?? themeName;

  return (
    <div ref={wrapperRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "inline-flex h-9 items-center gap-2 rounded-lg border border-border bg-card px-2.5 py-1.5",
          "text-xs font-medium text-foreground shadow-sm transition-colors hover:bg-secondary",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        )}
        title={t.theme?.switchTheme ?? "Switch theme"}
        aria-label={t.theme?.switchTheme ?? "Switch theme"}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <Palette className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="hidden sm:inline">{label}</span>
      </button>

      {open && (
        <div
          role="listbox"
          aria-label={t.theme?.title ?? "Theme"}
          className={cn(
            "absolute right-0 top-full z-50 mt-2 min-w-[260px] overflow-hidden rounded-xl",
            "border border-border bg-background shadow-xl backdrop-blur-xl",
          )}
        >
          <div className="border-b border-border px-3 py-2 text-xs font-semibold text-muted-foreground">
            {t.theme?.title ?? "Theme"}
          </div>

          {availableThemes.map((th) => {
            const isActive = th.name === themeName;
            const preset = BUILTIN_THEMES[th.name];

            return (
              <button
                key={th.name}
                type="button"
                role="option"
                aria-selected={isActive}
                onClick={() => {
                  setTheme(th.name);
                  close();
                }}
                className={cn(
                  "flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors cursor-pointer",
                  "hover:bg-secondary",
                  isActive ? "bg-secondary text-foreground" : "text-foreground",
                )}
              >
                {preset ? (
                  <ThemeSwatch theme={preset.name} />
                ) : (
                  <PlaceholderSwatch />
                )}

                <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <div className="truncate text-sm font-medium">{th.label}</div>
                  {th.description && (
                    <div className="truncate text-xs text-muted-foreground">
                      {th.description}
                    </div>
                  )}
                </div>

                <Check
                  className={cn(
                    "h-3.5 w-3.5 shrink-0 text-primary",
                    isActive ? "opacity-100" : "opacity-0",
                  )}
                />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ThemeSwatch({ theme }: { theme: string }) {
  const preset = BUILTIN_THEMES[theme];
  if (!preset) return <PlaceholderSwatch />;
  const { background, midground, warmGlow } = preset.palette;
  return (
    <div
      aria-hidden
      className="flex h-4 w-9 shrink-0 overflow-hidden border border-current/20"
    >
      <span className="flex-1" style={{ background: background.hex }} />
      <span className="flex-1" style={{ background: midground.hex }} />
      <span className="flex-1" style={{ background: warmGlow }} />
    </div>
  );
}

function PlaceholderSwatch() {
  return (
    <div
      aria-hidden
      className="h-4 w-9 shrink-0 border border-dashed border-current/20"
    />
  );
}
