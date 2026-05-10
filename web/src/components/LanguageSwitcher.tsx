import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";
import type { Locale } from "@/i18n/types";

const FLAGS: Record<Locale, { flag: string; label: string; next: Locale }> = {
  en: { flag: "🇬🇧", label: "EN", next: "zh" },
  zh: { flag: "🇨🇳", label: "中文", next: "es" },
  es: { flag: "🇪🇸", label: "ES", next: "en" },
};

/**
 * Compact language toggle — cycles through English, Chinese, and Spanish.
 * Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const toggle = () => setLocale(FLAGS[locale].next);

  return (
    <Button
      ghost
      onClick={toggle}
      title={t.language.switchTo}
      aria-label={t.language.switchTo}
      className="px-2 py-1 normal-case tracking-normal font-normal text-xs text-muted-foreground hover:text-foreground"
    >
      <span className="inline-flex items-center gap-1.5">
        <span className="text-base leading-none">
          {FLAGS[locale].flag}
        </span>

        <Typography
          mondwest
          className="hidden sm:inline tracking-wide uppercase text-[0.65rem]"
        >
          {FLAGS[locale].label}
        </Typography>
      </span>
    </Button>
  );
}
