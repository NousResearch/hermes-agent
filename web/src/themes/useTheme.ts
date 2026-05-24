import { useContext } from "react";
import { ThemeContext, type ThemeContextValue } from "./themeContext";

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
