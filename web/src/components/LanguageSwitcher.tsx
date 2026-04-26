import { useI18n } from "@/i18n/context";

/** Compact language toggle. Shows the current language and persists choice. */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const toggle = () => setLocale(locale === "en" ? "zh" : "en");

  return (
    <button
      type="button"
      onClick={toggle}
      className="inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-xs font-medium text-muted-foreground transition-colors cursor-pointer hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      title={t.language.switchTo}
      aria-label={t.language.switchTo}
    >
      <span className="text-base leading-none">
        {locale === "en" ? "🇬🇧" : "🇨🇳"}
      </span>
      <span className="hidden sm:inline">
        {locale === "en" ? "EN" : "中文"}
      </span>
    </button>
  );
}
