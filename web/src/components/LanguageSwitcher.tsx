import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";
import type { Locale } from "@/i18n";

/**
 * Compact language toggle — cycles English, Chinese, and Japanese.
 * Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const LOCALES: Locale[] = ["en", "zh", "ja"];
  const LABELS: Record<Locale, { flag: string; text: string }> = {
    en: { flag: "🇬🇧", text: "EN" },
    zh: { flag: "🇨🇳", text: "中文" },
    ja: { flag: "🇯🇵", text: "日本語" },
  };

  const toggle = () => {
    const current = LOCALES.indexOf(locale);
    setLocale(LOCALES[(current + 1) % LOCALES.length]);
  };

  const current = LABELS[locale];

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
          {current.flag}
        </span>

        <Typography
          mondwest
          className="hidden sm:inline tracking-wide uppercase text-[0.65rem]"
        >
          {current.text}
        </Typography>
      </span>
    </Button>
  );
}
