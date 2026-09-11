import { useI18n } from "@/i18n/context";
import { LOCALE_META } from "@/i18n";
import type { Locale } from "@/i18n";
import { cn } from "@/lib/utils";

/**
 * Language picker — a native <select>. Persists choice to localStorage via
 * the I18n context.
 *
 * 2026-09-02: rewritten from a hand-rolled role="listbox"/role="option"
 * dropdown (custom open/close state, manual arrow-key handling, a mobile
 * bottom-sheet variant) to a plain native <select>. The custom version kept
 * being reported as hard to use with a screen reader — even after adding
 * explicit ArrowUp/ArrowDown/Home/End handling, moving between options
 * didn't reliably announce which option was now focused. A native <select>
 * hands option navigation and announcement to the OS/browser's own combobox
 * implementation, which every screen reader already knows how to drive
 * correctly — no custom keyboard code needed, and it also collapses the
 * separate mobile bottom-sheet code path since native <select> already
 * renders as a proper picker on mobile.
 *
 * Ships 16 locales (en, zh, zh-hant, ja, de, es, fr, tr, uk, af, ko, it, ga,
 * pt, ru, hu). No country flags by design — languages aren't countries, and
 * flag pairings inevitably create political mismappings (e.g. Mandarin
 * variants ≠ any single jurisdiction, English ≠ GB, Portuguese ≠ PT).
 * Endonyms (each language's own name for itself) are unambiguous.
 */
export function LanguageSwitcher({ collapsed = false }: LanguageSwitcherProps) {
  const { locale, setLocale, t } = useI18n();
  const current = LOCALE_META[locale];
  const allLocales = Object.entries(LOCALE_META) as Array<[Locale, typeof current]>;

  return (
    <select
      aria-label={t.language.switchTo}
      title={t.language.switchTo}
      value={locale}
      onChange={(e) => setLocale(e.target.value as Locale)}
      className={cn(
        "px-2 py-1 normal-case tracking-normal font-normal text-xs text-text-secondary hover:text-foreground",
        "bg-transparent border border-transparent hover:border-border rounded cursor-pointer",
        collapsed && "hover:bg-transparent",
      )}
    >
      {allLocales.map(([code, meta]) => (
        <option key={code} value={code}>
          {meta.name}
        </option>
      ))}
    </select>
  );
}

interface LanguageSwitcherProps {
  collapsed?: boolean;
}
