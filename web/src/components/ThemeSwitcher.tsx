import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Palette, Check, Type } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { BottomSheet } from "@nous-research/ui/ui/components/bottom-sheet";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { useBelowBreakpoint } from "@nous-research/ui/hooks/use-below-breakpoint";
import { BUILTIN_THEMES, THEME_DEFAULT_FONT_ID, useTheme } from "@/themes";
import type { DashboardTheme, FontChoice, ThemeListEntry } from "@/themes";
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
export function ThemeSwitcher({
  collapsed = false,
  dropUp = false,
}: ThemeSwitcherProps) {
  const { themeName, availableThemes, setTheme, fontId, fontChoices, setFont } =
    useTheme();
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [dropUpStyle, setDropUpStyle] = useState<
    { bottom: number; left: number } | undefined
  >();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
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
      const target = e.target as Node;
      if (wrapperRef.current?.contains(target)) return;
      if (dropdownRef.current?.contains(target)) return;
      close();
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [open, close, useMobileSheet]);

  const current = availableThemes.find((th) => th.name === themeName);
  const label = current?.label ?? themeName;
  const sheetTitle = t.theme.title;
  const toggleOpen = () => {
    if (!open && dropUp && !useMobileSheet) {
      const rect = wrapperRef.current?.getBoundingClientRect();
      setDropUpStyle(
        rect
          ? { bottom: window.innerHeight - rect.top + 4, left: rect.left }
          : undefined,
      );
    }
    setOpen((value) => !value);
  };

  return (
    <div ref={wrapperRef} className="relative">
      <Button
        ghost
        size={collapsed ? "icon" : undefined}
        onClick={toggleOpen}
        className={cn(
          collapsed
            ? "text-text-secondary hover:text-foreground hover:bg-transparent"
            : "px-2 py-1 normal-case tracking-normal font-normal text-xs text-text-secondary hover:text-foreground",
        )}
        title={`${t.theme.switchTheme}: ${label}`}
        aria-label={t.theme.switchTheme}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className="inline-flex items-center gap-1.5">
          <Palette className="h-3.5 w-3.5" />

          {!collapsed && (
            <Typography className="hidden sm:inline text-display tracking-wide text-xs">
              {label}
            </Typography>
          )}
        </span>
      </Button>

      {useMobileSheet && (
        <BottomSheet
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
              t={t}
              themeName={themeName}
            />
            <FontSection
              fontChoices={fontChoices}
              fontId={fontId}
              setFont={setFont}
            />
          </div>
        </BottomSheet>
      )}

      {open &&
        !useMobileSheet &&
        (() => {
          const dropdown = (
            <div
              ref={dropdownRef}
              aria-label={sheetTitle}
              className={cn(
                "min-w-[240px] max-h-[70dvh] overflow-y-auto",
                "border border-current/20 bg-background-base/95",
                "shadow-[0_12px_32px_-8px_rgba(0,0,0,0.6)]",
                dropUp
                  ? "fixed z-[100]"
                  : "absolute z-50 right-0 top-full mt-1",
              )}
              role="listbox"
              style={dropUp ? dropUpStyle : undefined}
            >
              <div className="border-b border-current/20 px-3 py-2">
                <Typography className="text-display text-xs tracking-[0.12em] text-text-tertiary">
                  {sheetTitle}
                </Typography>
              </div>

              <ThemeSwitcherOptions
                availableThemes={availableThemes}
                close={close}
                setTheme={setTheme}
                t={t}
                themeName={themeName}
              />
              <FontSection
                fontChoices={fontChoices}
                fontId={fontId}
                setFont={setFont}
              />
            </div>
          );
          return dropUp ? createPortal(dropdown, document.body) : dropdown;
        })()}
    </div>
  );
}

function ThemeSwitcherOptions({
  availableThemes,
  close,
  setTheme,
  t,
  themeName,
}: ThemeSwitcherOptionsProps) {
  return (
    <>
      {availableThemes.map((th) => {
        const isActive = th.name === themeName;
        const paletteTheme = BUILTIN_THEMES[th.name] ?? th.definition;

        return (
          <ListItem
            active={isActive}
            aria-selected={isActive}
            className="gap-3"
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
              <Typography className="truncate text-display text-xs tracking-wide">
                {t.theme.presets[th.name]?.label ?? th.label}
              </Typography>
              {th.description && (
                <Typography className="truncate text-xs tracking-normal text-text-tertiary">
                  {t.theme.presets[th.name]?.description ?? th.description}
                </Typography>
              )}
            </div>

            <Check
              className={cn(
                "h-3 w-3 shrink-0 text-midground",
                isActive ? "opacity-100" : "opacity-0",
              )}
            />
          </ListItem>
        );
      })}
    </>
  );
}

const FONT_CATEGORY_LABEL_KEY: Record<
  FontChoice["category"],
  "fontSans" | "fontSerif" | "fontMono"
> = {
  sans: "fontSans",
  serif: "fontSerif",
  mono: "fontMono",
};

/** Font-override section rendered below the theme list. Lets the user pick
 *  any catalog font independently of the active theme, or "Theme default"
 *  to clear the override. Each row previews itself in its own font. */
function FontSection({ fontChoices, fontId, setFont }: FontSectionProps) {
  const { t } = useI18n();
  const order: FontChoice["category"][] = ["sans", "serif", "mono"];
  return (
    <>
      <div className="mt-1 border-t border-current/20 px-3 pb-1 pt-2">
        <span className="inline-flex items-center gap-1.5">
          <Type className="h-3 w-3 text-text-tertiary" />
          <Typography className="text-display text-xs tracking-[0.12em] text-text-tertiary">
            {t.theme.fontTitle}
          </Typography>
        </span>
      </div>

      {/* Theme-default (clears the override). */}
      <ListItem
        active={fontId === THEME_DEFAULT_FONT_ID}
        aria-selected={fontId === THEME_DEFAULT_FONT_ID}
        className="gap-3"
        onClick={() => setFont(THEME_DEFAULT_FONT_ID)}
        role="option"
      >
        <span aria-hidden className="h-4 w-9 shrink-0" />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <Typography className="truncate text-xs tracking-normal">
            {t.theme.fontDefault}
          </Typography>
          <Typography className="truncate text-xs tracking-normal text-text-tertiary">
            {t.theme.fontDefaultHint}
          </Typography>
        </div>
        <Check
          className={cn(
            "h-3 w-3 shrink-0 text-midground",
            fontId === THEME_DEFAULT_FONT_ID ? "opacity-100" : "opacity-0",
          )}
        />
      </ListItem>

      {order.map((cat) => {
        const fonts = fontChoices.filter((f) => f.category === cat);
        if (fonts.length === 0) return null;
        const catLabel = t.theme?.[FONT_CATEGORY_LABEL_KEY[cat]] ?? cat;
        return (
          <div key={cat}>
            <div className="px-3 pb-0.5 pt-1.5">
              <Typography className="text-[0.65rem] uppercase tracking-[0.1em] text-text-tertiary">
                {catLabel}
              </Typography>
            </div>
            {fonts.map((f) => {
              const isActive = f.id === fontId;
              return (
                <ListItem
                  active={isActive}
                  aria-selected={isActive}
                  className="gap-3"
                  key={f.id}
                  onClick={() => setFont(f.id)}
                  role="option"
                >
                  <span aria-hidden className="h-4 w-9 shrink-0" />
                  <div className="flex min-w-0 flex-1 flex-col">
                    {/* Preview the font in its own stack. */}
                    <span
                      className="truncate text-sm"
                      style={{ fontFamily: f.stack }}
                    >
                      {f.label}
                    </span>
                  </div>
                  <Check
                    className={cn(
                      "h-3 w-3 shrink-0 text-midground",
                      isActive ? "opacity-100" : "opacity-0",
                    )}
                  />
                </ListItem>
              );
            })}
          </div>
        );
      })}
    </>
  );
}

function ThemeSwatch({ theme }: { theme: DashboardTheme }) {
  const [c1, c2, c3] = theme.swatchColors ?? [
    theme.palette.background.hex,
    theme.palette.midground.hex,
    theme.palette.warmGlow,
  ];
  return (
    <div
      aria-hidden
      className="flex h-4 w-9 shrink-0 overflow-hidden border border-current/20"
    >
      <span className="flex-1" style={{ background: c1 }} />
      <span className="flex-1" style={{ background: c2 }} />
      <span className="flex-1" style={{ background: c3 }} />
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

interface ThemeSwitcherOptionsProps {
  availableThemes: ThemeListEntry[];
  close: () => void;
  setTheme: (name: string) => void;
  t: ReturnType<typeof useI18n>["t"];
  themeName: string;
}

interface FontSectionProps {
  fontChoices: FontChoice[];
  fontId: string;
  setFont: (id: string) => void;
}

interface ThemeSwitcherProps {
  collapsed?: boolean;
  dropUp?: boolean;
}
