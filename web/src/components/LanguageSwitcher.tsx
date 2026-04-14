import { useTranslation } from "react-i18next";
import { STORAGE_KEY } from "@/i18n";

export function LanguageSwitcher() {
  const { i18n } = useTranslation();

  const toggle = () => {
    const next = i18n.language === "zh" ? "en" : "zh";
    i18n.changeLanguage(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
  };

  return (
    <button
      type="button"
      onClick={toggle}
      className="font-display text-[0.7rem] tracking-[0.12em] uppercase text-muted-foreground hover:text-foreground transition-colors cursor-pointer px-2 py-1"
      title={i18n.language === "zh" ? "Switch to English" : "切换到中文"}
    >
      {i18n.language === "zh" ? "EN" : "中文"}
    </button>
  );
}
