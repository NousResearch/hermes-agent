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
      className="inline-flex h-9 items-center gap-2 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs font-medium text-foreground shadow-sm transition-colors hover:bg-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      title={t.language.switchTo}
      aria-label={t.language.switchTo}
    >
      <span className="text-base leading-none">
        {locale === "en" ? "🇬🇧" : "🇨🇳"}
      </span>
      <span className="hidden sm:inline">{locale === "en" ? "EN" : "中文"}</span>
    </button>
  );
}
