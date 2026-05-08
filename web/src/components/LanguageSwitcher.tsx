import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";
import { SUPPORTED_LOCALES, type Locale } from "@/i18n/types";

/**
 * Compact language toggle. Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();
  const options: Record<Locale, { displayName: string; flag: string }> = {
    en: { displayName: "English", flag: "🇬🇧" },
    zh: { displayName: "中文", flag: "🇨🇳" },
    "pt-BR": { displayName: "Português (Brasil)", flag: "🇧🇷" },
  };

  const currentIndex = SUPPORTED_LOCALES.indexOf(locale);
  const nextLocale = SUPPORTED_LOCALES[(currentIndex + 1) % SUPPORTED_LOCALES.length];
  const current = options[locale];
  const next = options[nextLocale];
  const label = `${t.language.switchTo}: ${next.displayName}`;

  const toggle = () => setLocale(nextLocale);

  return (
    <Button
      ghost
      onClick={toggle}
      title={label}
      aria-label={label}
      className="px-2 py-1 normal-case tracking-normal font-normal text-xs text-muted-foreground hover:text-foreground"
    >
      <span className="inline-flex items-center gap-1.5">
        <span className="text-base leading-none">
          {current.flag}
        </span>

        <Typography
          mondwest
          className="hidden sm:inline tracking-wide whitespace-nowrap text-[0.65rem]"
        >
          {current.displayName}
        </Typography>
      </span>
    </Button>
  );
}
