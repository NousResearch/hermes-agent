import { useState, useRef, useEffect } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";
import { LOCALE_META } from "@/i18n";
import type { Locale } from "@/i18n";

/**
 * Language picker — shows the current language's flag + endonym, opens a
 * dropdown of all supported locales when clicked.  Persists choice to
 * localStorage via the I18n context.
 *
 * Replaces the older two-state EN↔ZH toggle now that we ship 16 locales
 * (en, zh, zh-hant, ja, de, es, fr, tr, uk, af, ko, it, ga, pt, ru, hu).
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close on outside click / Escape so the dropdown doesn't trap the user.
  useEffect(() => {
    if (!open) return;

    function onPointerDown(e: PointerEvent) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const current = LOCALE_META[locale];
  const allLocales = Object.entries(LOCALE_META) as Array<[Locale, typeof current]>;

  return (
    <div ref={containerRef} className="relative inline-flex">
      <Button
        ghost
        onClick={() => setOpen((v) => !v)}
        title={t.language.switchTo}
        aria-label={t.language.switchTo}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="px-2 py-1 normal-case tracking-normal font-normal text-xs text-muted-foreground hover:text-foreground"
      >
        <span className="inline-flex items-center gap-1.5">
          <span className="text-base leading-none">{current.flag}</span>
          <Typography
            mondwest
            className="hidden sm:inline tracking-wide uppercase text-[0.65rem]"
          >
            {locale === "en" ? "EN" : current.name}
          </Typography>
        </span>
      </Button>

      {open && (
        <div
          role="listbox"
          aria-label={t.language.switchTo}
          className="absolute right-0 top-full mt-1 z-50 min-w-[10rem] rounded-md border border-border bg-popover shadow-md py-1 max-h-80 overflow-y-auto"
        >
          {allLocales.map(([code, meta]) => {
            const selected = code === locale;
            return (
              <button
                key={code}
                role="option"
                aria-selected={selected}
                onClick={() => {
                  setLocale(code);
                  setOpen(false);
                }}
                className={
                  "w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 hover:bg-accent hover:text-accent-foreground transition-colors " +
                  (selected ? "font-semibold text-foreground" : "text-muted-foreground")
                }
              >
                <span className="text-base leading-none">{meta.flag}</span>
                <span className="truncate">{meta.name}</span>
                {selected && <span className="ml-auto text-xs">✓</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
