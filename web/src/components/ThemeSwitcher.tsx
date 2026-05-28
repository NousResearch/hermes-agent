import { useCallback, useEffect, useRef, useState } from "react";
import { Palette, Check } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { BottomPickSheet } from "@/components/BottomPickSheet";
import { Typography } from "@/components/NouiTypography";
import { useBelowBreakpoint } from "@/hooks/useBelowBreakpoint";
import { BUILTIN_THEMES, useTheme } from "@/themes";
import type { DashboardTheme, ThemeListEntry } from "@/themes";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * Compact theme picker mounted next to the language switcher in the header.
 * Each dropdown row shows a 3-stop swatch (background / midground / warm
 * glow) so users can preview the palette before committing. User-defined
 * themes from `~/.hermes/dashboard-themes/*.yaml` use their API-provided
 * definitions so they show real palette swatches just like built-ins.
 *
 * When placed at the bottom of a container (e.g. the sidebar rail), pass
 * `dropUp` so the menu opens above the trigger instead of clipping below
 * the viewport. On viewports below the `sm` breakpoint, `dropUp` uses a
 * bottom sheet portaled to `document.body` so the picker is not clipped by
 * the sidebar (same idea as a responsive Drawer).
 */
export function ThemeSwitcher({ dropUp = false }: ThemeSwitcherProps) {
  const { themeName, availableThemes, setTheme } = useTheme();
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const narrowViewport = useBelowBreakpoint(640);
  const useMobileSheet = Boolean(dropUp && narrowViewport);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  useEffect(() => {
    if (!open || useMobileSheet) return;
    const onMouseDown = (e: MouseEvent) => {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(e.target as Node)
      ) {
        close();
      }
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [open, close, useMobileSheet]);

  const current = availableThemes.find((th) => th.name === themeName);
  const label = BUILTIN_THEMES[themeName]?.label ?? current?.label ?? themeName;
  const sheetTitle = t.theme?.title ?? "Theme";

  return (
    <div ref={wrapperRef} className="relative">
      <Button
        ghost
        onClick={() => setOpen((o) => !o)}
        className="px-2 py-1 normal-case tracking-normal font-normal text-xs text-text-secondary hover:text-foreground"
        title={t.theme?.switchTheme ?? "Switch theme"}
        aria-label={t.theme?.switchTheme ?? "Switch theme"}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className="inline-flex items-center gap-1.5">
          <Palette className="h-3.5 w-3.5" />

          <Typography
            className="hidden text-xs font-medium tracking-normal sm:inline"
          >
            {label}
          </Typography>
        </span>
      </Button>

      {useMobileSheet && (
        <BottomPickSheet
          backdropDismissLabel={t.common.close}
          onClose={close}
          open={open}
          title={sheetTitle}
        >
          <div aria-label={sheetTitle} role="listbox">
            <ThemeSwitcherOptions
              availableThemes={availableThemes}
              close={close}
              setTheme={setTheme}
              themeName={themeName}
            />
          </div>
        </BottomPickSheet>
      )}

      {open && !useMobileSheet && (
        <div
          aria-label={sheetTitle}
          className={cn(
            "absolute z-50 min-w-[240px] max-h-[70dvh] overflow-y-auto",
            dropUp ? "left-0 bottom-full mb-1" : "right-0 top-full mt-1",
            "rounded-lg border border-border bg-popover p-1",
            "shadow-[0_8px_24px_rgba(31,35,41,0.1),0_1px_2px_rgba(31,35,41,0.06)]",
          )}
          role="listbox"
        >
          <div className="border-b border-border px-3 py-2">
            <Typography
              className="text-xs font-medium tracking-normal text-text-tertiary"
            >
              {sheetTitle}
            </Typography>
          </div>

          <ThemeSwitcherOptions
            availableThemes={availableThemes}
            close={close}
            setTheme={setTheme}
            themeName={themeName}
          />
        </div>
      )}
    </div>
  );
}

function ThemeSwitcherOptions({
  availableThemes,
  close,
  setTheme,
  themeName,
}: ThemeSwitcherOptionsProps) {
  return (
    <>
      {availableThemes.map((th) => {
        const isActive = th.name === themeName;
        const builtinTheme = BUILTIN_THEMES[th.name];
        const paletteTheme = builtinTheme ?? th.definition;
        const label = builtinTheme?.label ?? th.label;
        const description = builtinTheme?.description ?? th.description;

        return (
          <ListItem
            active={isActive}
            aria-selected={isActive}
            className="gap-3 rounded-md px-2.5 py-2"
            key={th.name}
            onClick={() => {
              setTheme(th.name);
              close();
            }}
            role="option"
          >
            {paletteTheme ? (
              <ThemeSwatch theme={paletteTheme} />
            ) : (
              <PlaceholderSwatch />
            )}

            <div className="flex min-w-0 flex-1 flex-col gap-0.5">
              <Typography
                className="truncate text-xs font-medium tracking-normal"
              >
                {label}
              </Typography>
              {description && (
                <Typography className="truncate text-xs tracking-normal text-text-tertiary">
                  {description}
                </Typography>
              )}
            </div>

            <Check
              className={cn(
                "h-3 w-3 shrink-0 text-primary",
                isActive ? "opacity-100" : "opacity-0",
              )}
            />
          </ListItem>
        );
      })}
    </>
  );
}

function ThemeSwatch({ theme }: { theme: DashboardTheme }) {
  const { background, midground, warmGlow } = theme.palette;
  return (
    <div
      aria-hidden
      className="flex h-4 w-9 shrink-0 overflow-hidden rounded border border-border"
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
      className="h-4 w-9 shrink-0 rounded border border-dashed border-border"
    />
  );
}

interface ThemeSwitcherOptionsProps {
  availableThemes: ThemeListEntry[];
  close: () => void;
  setTheme: (name: string) => void;
  themeName: string;
}

interface ThemeSwitcherProps {
  dropUp?: boolean;
}
