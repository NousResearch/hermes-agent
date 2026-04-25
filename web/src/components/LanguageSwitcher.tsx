import { Typography } from "@nous-research/ui";
import { useI18n } from "@/i18n/context";
import type { Locale } from "@/i18n";

const LOCALE_CONFIG: Record<Locale, { flag: string; label: string }> = {                                                                                                                              
    en: { flag: "🇬🇧", label: "EN" },                                                                                                                                                                    
    zh: { flag: "🇨🇳", label: "中文" },                                                                                                                                                                  
    pt: { flag: "🇧🇷", label: "PT" },                                                                                                                                                                    
  };     
         
/**
 * Compact language toggle — shows a clickable flag that switches between
 * English and Chinese.  Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const locales = Object.keys(LOCALE_CONFIG) as Locale[];
  const cycle = () => {
    const next = locales[(locales.indexOf(locale) + 1) % locales.length];
    setLocale(next);
  };

  const current = LOCALE_CONFIG[locale];

  return (
    <button
      type="button"
      onClick={cycle}
      className="group relative inline-flex items-center gap-1.5 px-2 py-1 text-xs text-muted-foreground hover:text-foreground transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      title={t.language.switchTo}
      aria-label={t.language.switchTo}
    >
      <span className="text-base leading-none">
        {current.flag}
      </span>
      <Typography
        mondwest
        className="hidden sm:inline tracking-wide uppercase text-[0.65rem]"
      >
        {current.label}
      </Typography>
    </button>
  );
}
