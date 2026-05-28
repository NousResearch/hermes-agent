import { useState, useRef, useEffect } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { BottomPickSheet } from "@/components/BottomPickSheet";
import { Typography } from "@/components/NouiTypography";
import { useBelowBreakpoint } from "@/hooks/useBelowBreakpoint";
import { useI18n } from "@/i18n/context";
import { LOCALE_META } from "@/i18n";
import type { Locale } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * Language picker — shows the current language's endonym, opens a dropdown
 * of all supported locales when clicked.  Persists choice to localStorage via
 * the I18n context.
 *
 * Replaces the older two-state EN↔ZH toggle now that we ship 16 locales
 * (en, zh, zh-hant, ja, de, es, fr, tr, uk, af, ko, it, ga, pt, ru, hu).
 *
 * No country flags by design — languages aren't countries, and flag pairings
 * inevitably create political mismappings (e.g. Mandarin variants ≠ any single
 * jurisdiction, English ≠ GB, Portuguese ≠ PT). Endonyms are unambiguous.
 *
 * When placed at the bottom of the sidebar (next to ThemeSwitcher), pass
 * `dropUp` so the list opens above the trigger and avoids clipping below the
 * viewport / overflow ancestors. Below the `sm` breakpoint, `dropUp` uses a
 * bottom sheet portaled to `document.body` instead of an anchored dropdown.
 */
export function LanguageSwitcher({ dropUp = false }: LanguageSwitcherProps) {
  const { locale, setLocale, t } = useI18n();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const narrowViewport = useBelowBreakpoint(640);
  const useMobileSheet = Boolean(dropUp && narrowViewport);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  // Outside-click closing only for anchored dropdown — sheet uses backdrop + portal.
  useEffect(() => {
    if (!open || useMobileSheet) return;

    function onPointerDown(e: PointerEvent) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }

    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open, useMobileSheet]);

  const current = LOCALE_META[locale];
  const allLocales = Object.entries(LOCALE_META) as Array<[Locale, typeof current]>;
  const sheetTitle = t.language.switchTo;

  return (
    <div ref={containerRef} className="relative inline-flex">
      <Button
        ghost
        onClick={() => setOpen((v) => !v)}
        title={t.language.switchTo}
        aria-label={t.language.switchTo}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="px-2 py-1 normal-case tracking-normal font-normal text-xs text-text-secondary hover:text-foreground"
      >
        <span className="inline-flex items-center gap-1.5">
          <Typography
            mondwest
            className="hidden sm:inline text-xs font-medium tracking-normal"
          >
            {locale === "en" ? "EN" : current.name}
          </Typography>
        </span>
      </Button>

      {useMobileSheet && (
        <BottomPickSheet
          backdropDismissLabel={t.common.close}
          onClose={() => setOpen(false)}
          open={open}
          title={sheetTitle}
        >
          <div aria-label={sheetTitle} role="listbox">
            <LanguageSwitcherOptions
              allLocales={allLocales}
              locale={locale}
              setLocale={setLocale}
              setOpen={setOpen}
            />
          </div>
        </BottomPickSheet>
      )}

      {open && !useMobileSheet && (
        <div
          aria-label={sheetTitle}
          className={cn(
            "absolute right-0 z-50 min-w-[11rem] rounded-lg border border-border bg-popover p-1 shadow-[0_8px_24px_rgba(31,35,41,0.1),0_1px_2px_rgba(31,35,41,0.06)] max-h-80 overflow-y-auto",
            dropUp ? "bottom-full mb-1" : "top-full mt-1",
          )}
          role="listbox"
        >
          <LanguageSwitcherOptions
            allLocales={allLocales}
            locale={locale}
            setLocale={setLocale}
            setOpen={setOpen}
          />
        </div>
      )}
    </div>
  );
}

function LanguageSwitcherOptions({
  allLocales,
  locale,
  setLocale,
  setOpen,
}: LanguageSwitcherOptionsProps) {
  return (
    <>
      {allLocales.map(([code, meta]) => {
        const selected = code === locale;

        return (
          <button
            aria-selected={selected}
            className={
              "w-full min-h-8 rounded-md text-left px-3 py-1.5 text-xs flex items-center gap-2 hover:bg-muted hover:text-foreground transition-colors " +
              (selected ? "font-medium bg-primary/5 text-primary" : "font-normal text-text-secondary")
            }
            key={code}
            onClick={() => {
              setLocale(code);
              setOpen(false);
            }}
            role="option"
            type="button"
          >
            <span className="truncate">{meta.name}</span>

            {selected && <span className="ml-auto text-xs">✓</span>}
          </button>
        );
      })}
    </>
  );
}

interface LanguageSwitcherOptionsProps {
  allLocales: Array<[Locale, (typeof LOCALE_META)[Locale]]>;
  locale: Locale;
  setLocale: (code: Locale) => void;
  setOpen: (open: boolean) => void;
}

interface LanguageSwitcherProps {
  dropUp?: boolean;
}
