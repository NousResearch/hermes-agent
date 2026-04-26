import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Palette } from "lucide-react";
import { BUILTIN_THEMES, useTheme } from "@/themes";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/** Theme picker mounted beside the language switcher in the sidebar footer. */
export function ThemeSwitcher({ dropUp = false, iconOnly = false }: ThemeSwitcherProps) {
  const { themeName, availableThemes, setTheme } = useTheme();
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
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
    <div ref={wrapperRef} className="relative min-w-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "inline-flex h-8 min-w-0 items-center gap-2 rounded-md px-2 text-xs font-medium text-muted-foreground transition-colors cursor-pointer",
          "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          iconOnly ? "size-9 justify-center px-0" : "max-w-full",
        )}
        title={t.theme?.switchTheme ?? "Switch theme"}
        aria-label={t.theme?.switchTheme ?? "Switch theme"}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <Palette className="h-4 w-4 shrink-0" />
        {!iconOnly && <span className="truncate">{label}</span>}
      </button>

      {open && (
        <div
          role="listbox"
          aria-label={t.theme?.title ?? "Theme"}
          className={cn(
            "absolute z-50 w-[min(24rem,calc(100vw-2rem))] overflow-hidden rounded-xl border border-border bg-popover text-popover-foreground shadow-xl",
            dropUp ? "left-0 bottom-full mb-2" : "right-0 top-full mt-2",
          )}
        >
          <div className="border-b border-border px-3 py-2 text-xs font-medium text-muted-foreground">
            {t.theme?.title ?? "Theme"}
          </div>

          <div className="max-h-[22rem] overflow-y-auto p-1">
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
                    "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors cursor-pointer",
                    isActive ? "bg-accent text-accent-foreground" : "hover:bg-accent/70 hover:text-accent-foreground",
                  )}
                >
                  {preset ? <ThemeSwatch theme={preset.name} /> : <PlaceholderSwatch />}

                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="truncate text-sm font-medium">{th.label}</span>
                    {th.description && (
                      <span className="truncate text-xs text-muted-foreground">
                        {th.description}
                      </span>
                    )}
                  </div>

                  <Check className={cn("h-4 w-4 shrink-0", isActive ? "opacity-100" : "opacity-0")} />
                </button>
              );
            })}
          </div>
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
    <div aria-hidden className="flex h-5 w-10 shrink-0 overflow-hidden rounded-md border border-border">
      <span className="flex-1" style={{ background: background.hex }} />
      <span className="flex-1" style={{ background: midground.hex }} />
      <span className="flex-1" style={{ background: warmGlow }} />
    </div>
  );
}

function PlaceholderSwatch() {
  return <div aria-hidden className="h-5 w-10 shrink-0 rounded-md border border-dashed border-border" />;
}

interface ThemeSwitcherProps {
  dropUp?: boolean;
  iconOnly?: boolean;
}
