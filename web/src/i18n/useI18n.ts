import { useContext } from "react";
import { I18nContext } from "./i18nContext";

export function useI18n() {
  return useContext(I18nContext);
}
