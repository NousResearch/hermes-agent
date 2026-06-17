import { useContext } from "react";
import { ThemeContext, type ThemeContextValue } from "./shared";

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
