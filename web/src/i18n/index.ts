import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./locales/en.json";
import zh from "./locales/zh.json";

type Language = "en" | "zh";

const STORAGE_KEY = "hermes-agent-language";
const DEFAULT_LANGUAGE: Language = "en";

const getInitialLanguage = (): Language => {
  if (typeof window !== "undefined") {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored === "en" || stored === "zh") return stored;
    } catch {
      // ignore
    }
  }

  const nav =
    typeof navigator !== "undefined"
      ? (navigator.language?.toLowerCase() ??
        navigator.languages?.[0]?.toLowerCase())
      : undefined;

  if (nav?.startsWith("zh")) return "zh";

  return DEFAULT_LANGUAGE;
};

i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    zh: { translation: zh },
  },
  lng: getInitialLanguage(),
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
  },
  debug: false,
});

export { STORAGE_KEY };
export default i18n;
