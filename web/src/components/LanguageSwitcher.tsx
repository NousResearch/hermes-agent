import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@/components/NouiTypography";
import { useI18n } from "@/i18n/context";

/**
 * Compact language toggle — shows a clickable flag that switches between
 * English, Simplified Chinese, and Traditional Chinese.  Persists choice to localStorage.
 */
export function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n();

  const toggle = () => {
    if (locale === "en") setLocale("zh");
    else if (locale === "zh") setLocale("tw");
    else setLocale("en");
  };

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
          {locale === "en" ? "🇬🇧" : locale === "zh" ? "🇨🇳" : "🇹🇼"}
        </span>
        <Typography
          mondwest
          className="hidden sm:inline tracking-wide uppercase text-[0.65rem]"
        >
          {locale === "en" ? "EN" : locale === "zh" ? "简体" : "繁體"}
        </Typography>
      </span>
    </Button>
  );
}
