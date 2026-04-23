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
import type {
  DashboardTheme,
  ThemeColorOverrides,
  ThemeDensity,
  ThemeLayer,
  ThemeLayout,
  ThemePalette,
  ThemeTypography,
} from "./types";
import { api } from "@/lib/api";

/** LocalStorage key — pre-applied before the React tree mounts to avoid
 *  a visible flash of the default palette on theme-overridden installs. */
const STORAGE_KEY = "hermes-dashboard-theme";

/** Tracks fontUrls we've already injected so multiple theme switches don't
 *  pile up <link> tags. Keyed by URL. */
const INJECTED_FONT_URLS = new Set<string>();

// ---------------------------------------------------------------------------
// CSS variable builders
// ---------------------------------------------------------------------------

/** Turn a ThemeLayer into the two CSS expressions the DS consumes:
 *  `--<name>` (color-mix'd with alpha) and `--<name>-base` (opaque hex). */
function layerVars(
  name: "background" | "midground" | "foreground",
  layer: ThemeLayer,
): Record<string, string> {
  const pct = Math.round(layer.alpha * 100);
  return {
    [`--${name}`]: `color-mix(in srgb, ${layer.hex} ${pct}%, transparent)`,
    [`--${name}-base`]: layer.hex,
    [`--${name}-alpha`]: String(layer.alpha),
  };
}

function paletteVars(palette: ThemePalette): Record<string, string> {
  return {
    ...layerVars("background", palette.background),
    ...layerVars("midground", palette.midground),
    ...layerVars("foreground", palette.foreground),
    "--warm-glow": palette.warmGlow,
    "--noise-opacity-mul": String(palette.noiseOpacity),
  };
}

const DENSITY_MULTIPLIERS: Record<ThemeDensity, string> = {
  compact: "0.85",
  comfortable: "1",
  spacious: "1.2",
};

function typographyVars(typo: ThemeTypography): Record<string, string> {
  return {
    "--theme-font-sans": typo.fontSans,
    "--theme-font-mono": typo.fontMono,
    "--theme-font-display": typo.fontDisplay ?? typo.fontSans,
    "--theme-base-size": typo.baseSize,
    "--theme-line-height": typo.lineHeight,
    "--theme-letter-spacing": typo.letterSpacing,
  };
}

function layoutVars(layout: ThemeLayout): Record<string, string> {
  return {
    "--radius": layout.radius,
    "--theme-radius": layout.radius,
    "--theme-spacing-mul": DENSITY_MULTIPLIERS[layout.density] ?? "1",
    "--theme-density": layout.density,
  };
}

/** Map a color-overrides key (camelCase) to its `--color-*` CSS var. */
const OVERRIDE_KEY_TO_VAR: Record<keyof ThemeColorOverrides, string> = {
  card: "--color-card",
  cardForeground: "--color-card-foreground",
  popover: "--color-popover",
  popoverForeground: "--color-popover-foreground",
  primary: "--color-primary",
  primaryForeground: "--color-primary-foreground",
  secondary: "--color-secondary",
  secondaryForeground: "--color-secondary-foreground",
  muted: "--color-muted",
  mutedForeground: "--color-muted-foreground",
  accent: "--color-accent",
  accentForeground: "--color-accent-foreground",
  destructive: "--color-destructive",
  destructiveForeground: "--color-destructive-foreground",
  success: "--color-success",
  warning: "--color-warning",
  border: "--color-border",
  input: "--color-input",
  ring: "--color-ring",
};

/** Keys we might have written on a previous theme — needed to know which
 *  properties to clear when a theme with fewer overrides replaces one
 *  with more. */
const ALL_OVERRIDE_VARS = Object.values(OVERRIDE_KEY_TO_VAR);

function overrideVars(
  overrides: ThemeColorOverrides | undefined,
): Record<string, string> {
  if (!overrides) return {};
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(overrides)) {
    if (!value) continue;
    const cssVar = OVERRIDE_KEY_TO_VAR[key as keyof ThemeColorOverrides];
    if (cssVar) out[cssVar] = value;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Font stylesheet injection
// ---------------------------------------------------------------------------

function injectFontStylesheet(url: string | undefined) {
  if (!url || typeof document === "undefined") return;
  if (INJECTED_FONT_URLS.has(url)) return;
  // Also skip if the page already has this href (e.g. SSR'd or persisted).
  const existing = document.querySelector<HTMLLinkElement>(
    `link[rel="stylesheet"][href="${CSS.escape(url)}"]`,
  );
  if (existing) {
    INJECTED_FONT_URLS.add(url);
    return;
  }
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = url;
  link.setAttribute("data-hermes-theme-font", "true");
  document.head.appendChild(link);
  INJECTED_FONT_URLS.add(url);
}

// ---------------------------------------------------------------------------
// Apply a full theme to :root
// ---------------------------------------------------------------------------

function applyTheme(theme: DashboardTheme) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;

  // Clear any overrides from a previous theme before applying the new set.
  for (const cssVar of ALL_OVERRIDE_VARS) {
    root.style.removeProperty(cssVar);
  }

  const vars = {
    ...paletteVars(theme.palette),
    ...typographyVars(theme.typography),
    ...layoutVars(theme.layout),
    ...overrideVars(theme.colorOverrides),
  };
  for (const [k, v] of Object.entries(vars)) {
    root.style.setProperty(k, v);
  }

  injectFontStylesheet(theme.typography.fontUrl);
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function ThemeProvider({ children }: { children: ReactNode }) {
  /** Name of the currently active theme (built-in id or user YAML name). */
  const [themeName, setThemeName] = useState<string>(() => {
    if (typeof window === "undefined") return "default";
    return window.localStorage.getItem(STORAGE_KEY) ?? "default";
  });

  /** All selectable themes (shown in the picker). Starts with just the
   *  built-ins; the API call below merges in user themes. */
  const [availableThemes, setAvailableThemes] = useState<
    Array<{ description: string; label: string; name: string }>
  >(() =>
    Object.values(BUILTIN_THEMES).map((t) => ({
      name: t.name,
      label: t.label,
      description: t.description,
    })),
  );

  /** Full definitions for user themes keyed by name — the API provides
   *  these so custom YAMLs apply without a client-side stub. */
  const [userThemeDefs, setUserThemeDefs] = useState<
    Record<string, DashboardTheme>
  >({});

  // Resolve a theme name to a full DashboardTheme, falling back to default
  // only when neither a built-in nor a user theme is found.
  const resolveTheme = useCallback(
    (name: string): DashboardTheme => {
      return (
        BUILTIN_THEMES[name] ??
        userThemeDefs[name] ??
        defaultTheme
      );
    },
    [userThemeDefs],
  );

  // Re-apply on every themeName change, or when user themes arrive from
  // the API (since the active theme might be a user theme whose definition
  // hadn't loaded yet on first render).
  useEffect(() => {
    applyTheme(resolveTheme(themeName));
  }, [themeName, resolveTheme]);

  // Load server-side themes (built-ins + user YAMLs) once on mount.
  useEffect(() => {
    let cancelled = false;
    api
      .getThemes()
      .then((resp) => {
        if (cancelled) return;
        if (resp.themes?.length) {
          setAvailableThemes(
            resp.themes.map((t) => ({
              name: t.name,
              label: t.label,
              description: t.description,
            })),
          );
          // Index any definitions the server shipped (user themes).
          const defs: Record<string, DashboardTheme> = {};
          for (const entry of resp.themes) {
            if (entry.definition) {
              defs[entry.name] = entry.definition;
            }
          }
          if (Object.keys(defs).length > 0) setUserThemeDefs(defs);
        }
        if (resp.active && resp.active !== themeName) {
          setThemeName(resp.active);
          window.localStorage.setItem(STORAGE_KEY, resp.active);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setTheme = useCallback(
    (name: string) => {
      // Accept any name the server told us exists OR any built-in.
      const knownNames = new Set<string>([
        ...Object.keys(BUILTIN_THEMES),
        ...availableThemes.map((t) => t.name),
        ...Object.keys(userThemeDefs),
      ]);
      const next = knownNames.has(name) ? name : "default";
      setThemeName(next);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(STORAGE_KEY, next);
      }
      api.setTheme(next).catch(() => {});
    },
    [availableThemes, userThemeDefs],
  );

  const value = useMemo<ThemeContextValue>(
    () => ({
      theme: resolveTheme(themeName),
      themeName,
      availableThemes,
      setTheme,
    }),
    [themeName, availableThemes, setTheme, resolveTheme],
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
