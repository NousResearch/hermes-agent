import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import type { DashboardTheme, ThemeColors, ThemeOverlay } from "./types";
import { BUILTIN_THEMES, defaultTheme } from "./presets";
import { api } from "@/lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Apply a theme's color overrides to `document.documentElement`. */
function applyColors(colors: ThemeColors) {
  const root = document.documentElement;
  for (const [key, value] of Object.entries(colors)) {
    root.style.setProperty(`--color-${key}`, value);
  }
}

/** Apply overlay overrides (noise + warm-glow). */
function applyOverlay(overlay: ThemeOverlay | undefined) {
  const noiseEl = document.querySelector<HTMLElement>(".noise-overlay");
  const glowEl = document.querySelector<HTMLElement>(".warm-glow");

  if (noiseEl) {
    noiseEl.style.opacity = String(overlay?.noiseOpacity ?? 0.10);
    noiseEl.style.mixBlendMode = overlay?.noiseBlendMode ?? "color-dodge";
  }
  if (glowEl) {
    glowEl.style.opacity = String(overlay?.warmGlowOpacity ?? 0.22);
    if (overlay?.warmGlowColor) {
      glowEl.style.background = `radial-gradient(ellipse at 0% 0%, ${overlay.warmGlowColor} 0%, rgba(0,0,0,0) 60%)`;
    }
  }
}

/** Remove all inline overrides — reverts to stylesheet defaults. */
function clearOverrides() {
  const root = document.documentElement;
  // Clear color overrides
  for (const key of Object.keys(defaultTheme.colors)) {
    root.style.removeProperty(`--color-${key}`);
  }
  // Clear overlay overrides
  const noiseEl = document.querySelector<HTMLElement>(".noise-overlay");
  const glowEl = document.querySelector<HTMLElement>(".warm-glow");
  if (noiseEl) {
    noiseEl.style.opacity = "";
    noiseEl.style.mixBlendMode = "";
  }
  if (glowEl) {
    glowEl.style.opacity = "";
    glowEl.style.background = "";
  }
}

function applyTheme(theme: DashboardTheme) {
  if (theme.name === "default") {
    clearOverrides();
  } else {
    applyColors(theme.colors);
    applyOverlay(theme.overlay);
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface ThemeContextValue {
  /** Currently active theme name. */
  themeName: string;
  /** Currently active theme object. */
  theme: DashboardTheme;
  /** Available theme names (built-in + any server-provided custom themes). */
  availableThemes: Array<{ name: string; label: string; description: string }>;
  /** Switch theme — applies CSS immediately and persists to config.yaml. */
  setTheme: (name: string) => void;
  /** True while initial theme is loading from server. */
  loading: boolean;
}

const ThemeContext = createContext<ThemeContextValue>({
  themeName: "default",
  theme: defaultTheme,
  availableThemes: Object.values(BUILTIN_THEMES).map((t) => ({
    name: t.name,
    label: t.label,
    description: t.description,
  })),
  setTheme: () => {},
  loading: true,
});

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [themeName, setThemeName] = useState("default");
  const [availableThemes, setAvailableThemes] = useState(
    Object.values(BUILTIN_THEMES).map((t) => ({
      name: t.name,
      label: t.label,
      description: t.description,
    })),
  );
  const [loading, setLoading] = useState(true);

  // Fetch active theme + available list from server on mount.
  useEffect(() => {
    api
      .getThemes()
      .then((resp) => {
        if (resp.themes?.length) {
          setAvailableThemes(resp.themes);
        }
        if (resp.active && resp.active !== "default") {
          setThemeName(resp.active);
          const t = BUILTIN_THEMES[resp.active];
          if (t) applyTheme(t);
        }
      })
      .catch(() => {
        // Server might not support theme API yet — stay on default.
      })
      .finally(() => setLoading(false));
  }, []);

  const resolvedTheme = BUILTIN_THEMES[themeName] ?? defaultTheme;

  const setTheme = useCallback(
    (name: string) => {
      const t = BUILTIN_THEMES[name] ?? defaultTheme;
      setThemeName(t.name);
      applyTheme(t);
      // Persist to config.yaml — fire and forget.
      api.setTheme(t.name).catch(() => {});
    },
    [],
  );

  return (
    <ThemeContext.Provider
      value={{
        themeName,
        theme: resolvedTheme,
        availableThemes,
        setTheme,
        loading,
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useTheme() {
  return useContext(ThemeContext);
}
