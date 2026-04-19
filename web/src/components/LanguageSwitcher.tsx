import { useI18n } from "@/i18n/context";

/**
 * Compact language toggle — shows a clickable flag that switches between
 * English and Chinese.  Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const toggle = () => setLocale(locale === "en" ? "zh" : "en");

  return (
    <button
      type="button"
      onClick={toggle}
      className="group relative inline-flex items-center gap-1.5 px-2 py-1 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      title={t.language.switchTo}
      aria-label={t.language.switchTo}
    >
      {/* Show the *current* language's flag — tooltip advertises the click action */}
      <span className="text-base leading-none">{locale === "en" ? "🇬🇧" : "🇨🇳"}</span>
      <span className="hidden sm:inline font-display tracking-wide uppercase text-[0.65rem]">
        {locale === "en" ? "EN" : "中文"}
      </span>
    </button>
  );
}
