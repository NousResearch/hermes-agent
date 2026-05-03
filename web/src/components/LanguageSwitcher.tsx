import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";
import type { Locale } from "@/i18n/types";

/**
 * Compact language toggle — cycles through supported locales.
 * Persists choice to localStorage. Uses flag emoji + short label.
 */
const ROTATION: Locale[] = ["en", "zh", "he"];

const FLAGS: Record<Locale, string> = {
  en: "🇬🇧",
  zh: "🇨🇳",
  he: "🇮🇱",
};

const LABELS: Record<Locale, string> = {
  en: "EN",
  zh: "中文",
  he: "עב",
};

export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const next = () => {
    const idx = ROTATION.indexOf(locale);
    const nextLocale = ROTATION[(idx + 1) % ROTATION.length];
    setLocale(nextLocale);
  };

  return (
    <Button
      ghost
      onClick={next}
      title={t.language.switchTo}
      aria-label={t.language.switchTo}
      className="px-2 py-1 normal-case tracking-normal font-normal text-xs text-muted-foreground hover:text-foreground"
    >
      <span className="inline-flex items-center gap-1.5">
        <span className="text-base leading-none">{FLAGS[locale]}</span>

        <Typography
          mondwest
          className="hidden sm:inline tracking-wide uppercase text-[0.65rem]"
        >
          {LABELS[locale]}
        </Typography>
      </span>
    </Button>
  );
}
