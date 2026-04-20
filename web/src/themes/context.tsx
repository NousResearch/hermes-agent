import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { BUILTIN_THEMES, defaultTheme } from "./presets";
import type { DashboardTheme, ThemeLayer, ThemePalette } from "./types";
import { api } from "@/lib/api";

/** LocalStorage key — pre-applied before the React tree mounts to avoid
 *  a visible flash of the default palette on theme-overridden installs. */
const STORAGE_KEY = "hermes-dashboard-theme";

/** Turn a ThemeLayer into the two CSS expressions the DS consumes:
 *  `--<name>` (color-mix'd with alpha) and `--<name>-base` (opaque hex). */
function layerVars(name: "background" | "midground" | "foreground", layer: ThemeLayer) {
  const pct = Math.round(layer.alpha * 100);
  return {
    [`--${name}`]: `color-mix(in srgb, ${layer.hex} ${pct}%, transparent)`,
    [`--${name}-base`]: layer.hex,
    [`--${name}-alpha`]: String(layer.alpha),
  };
}

/** Write a theme's palette to `document.documentElement` as inline styles.
 *  Inline styles beat the `:root { }` rule in index.css, so this cascades
 *  into every shadcn-compat token defined over the DS triplet. */
function applyPalette(palette: ThemePalette) {
  const root = document.documentElement;
  const vars = {
    ...layerVars("background", palette.background),
    ...layerVars("midground", palette.midground),
    ...layerVars("foreground", palette.foreground),
    "--warm-glow": palette.warmGlow,
    "--noise-opacity-mul": String(palette.noiseOpacity),
  };
  for (const [k, v] of Object.entries(vars)) {
    root.style.setProperty(k, v);
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [themeName, setThemeName] = useState<string>(() => {
    if (typeof window === "undefined") return "default";
    return window.localStorage.getItem(STORAGE_KEY) ?? "default";
  });
  const [availableThemes, setAvailableThemes] = useState<
    Array<{ description: string; label: string; name: string }>
  >(() =>
    Object.values(BUILTIN_THEMES).map((t) => ({
      name: t.name,
      label: t.label,
      description: t.description,
    })),
  );

  useEffect(() => {
    const t = BUILTIN_THEMES[themeName] ?? defaultTheme;
    applyPalette(t.palette);
  }, [themeName]);

  useEffect(() => {
    let cancelled = false;
    api
      .getThemes()
      .then((resp) => {
        if (cancelled) return;
        if (resp.themes?.length) setAvailableThemes(resp.themes);
        if (resp.active && resp.active !== themeName) {
          setThemeName(resp.active);
          window.localStorage.setItem(STORAGE_KEY, resp.active);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const setTheme = useCallback((name: string) => {
    const next = BUILTIN_THEMES[name] ? name : "default";
    setThemeName(next);
    window.localStorage.setItem(STORAGE_KEY, next);
    api.setTheme(next).catch(() => {});
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({
      theme: BUILTIN_THEMES[themeName] ?? defaultTheme,
      themeName,
      availableThemes,
      setTheme,
    }),
    [themeName, availableThemes, setTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: defaultTheme,
  themeName: "default",
  availableThemes: Object.values(BUILTIN_THEMES).map((t) => ({
    name: t.name,
    label: t.label,
    description: t.description,
  })),
  setTheme: () => {},
});

interface ThemeContextValue {
  availableThemes: Array<{ description: string; label: string; name: string }>;
  setTheme: (name: string) => void;
  theme: DashboardTheme;
  themeName: string;
}
