import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";
import type { Locale } from "@/i18n/types";

const LOCALE_CONFIG: Record<Locale, { flag: string; short: string }> = {
  en: { flag: "🇬🇧", short: "EN" },
  zh: { flag: "🇨🇳", short: "中文" },
  "pt-BR": { flag: "🇧🇷", short: "PT" },
};

const LOCALE_ORDER: Locale[] = ["en", "zh", "pt-BR"];

/**
 * Compact language toggle — shows a clickable flag that cycles through
 * supported locales.  Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const cycle = () => {
    const idx = LOCALE_ORDER.indexOf(locale);
    const next = LOCALE_ORDER[(idx + 1) % LOCALE_ORDER.length];
    setLocale(next);
  };

  const cfg = LOCALE_CONFIG[locale];

  return (
    <Button
      ghost
      onClick={cycle}
      title={t.language.switchTo}
      aria-label={t.language.switchTo}
      className="px-2 py-1 normal-case tracking-normal font-normal text-xs text-muted-foreground hover:text-foreground"
    >
      <span className="inline-flex items-center gap-1.5">
        <span className="text-base leading-none">{cfg.flag}</span>

        <Typography
          mondwest
          className="hidden sm:inline tracking-wide uppercase text-[0.65rem]"
        >
          {cfg.short}
        </Typography>
      </span>
    </Button>
  );
}
