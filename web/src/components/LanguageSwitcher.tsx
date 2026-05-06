import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";

/**
 * Compact language toggle — shows a clickable flag that switches between
 * English, Korean, and Chinese. Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const localeMeta = {
    en: { next: "ko", flag: "🇬🇧", label: "EN" },
    ko: { next: "zh", flag: "🇰🇷", label: "한국어" },
    zh: { next: "en", flag: "🇨🇳", label: "中文" },
  } as const;

  const current = localeMeta[locale];
  const toggle = () => setLocale(current.next);

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
          {current.label}
        </Typography>
      </span>
    </Button>
  );
}
