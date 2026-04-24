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
  ThemeAssets,
  ThemeColorOverrides,
  ThemeComponentStyles,
  ThemeDensity,
  ThemeLayer,
  ThemeLayout,
  ThemeLayoutVariant,
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
// Asset + component-style + layout variant vars
// ---------------------------------------------------------------------------

/** Well-known named asset slots a theme may populate. Kept in sync with
 *  `_THEME_NAMED_ASSET_KEYS` in `hermes_cli/web_server.py`. */
const NAMED_ASSET_KEYS = ["bg", "hero", "logo", "crest", "sidebar", "header"] as const;

/** Component buckets mirrored from the backend's `_THEME_COMPONENT_BUCKETS`.
 *  Each bucket emits `--component-<bucket>-<kebab-prop>` CSS vars. */
const COMPONENT_BUCKETS = [
  "card", "header", "footer", "sidebar", "tab",
  "progress", "badge", "backdrop", "page",
] as const;

/** Camel → kebab (`clipPath` → `clip-path`). */
function toKebab(s: string): string {
  return s.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
}

/** Build `--theme-asset-*` CSS vars from the assets block. Values are wrapped
 *  in `url(...)` when they look like a bare path/URL; raw CSS expressions
 *  (`linear-gradient(...)`, pre-wrapped `url(...)`, `none`) pass through. */
function assetVars(assets: ThemeAssets | undefined): Record<string, string> {
  if (!assets) return {};
  const out: Record<string, string> = {};
  const wrap = (v: string): string => {
    const trimmed = v.trim();
    if (!trimmed) return "";
    // Already a CSS image/gradient/url/none — don't re-wrap.
    if (/^(url\(|linear-gradient|radial-gradient|conic-gradient|none$)/i.test(trimmed)) {
      return trimmed;
    }
    // Bare path / http(s) URL / data: URL → wrap in url().
    return `url("${trimmed.replace(/"/g, '\\"')}")`;
  };
  for (const key of NAMED_ASSET_KEYS) {
    const val = assets[key];
    if (typeof val === "string" && val.trim()) {
      out[`--theme-asset-${key}`] = wrap(val);
      out[`--theme-asset-${key}-raw`] = val;
    }
  }
  if (assets.custom) {
    for (const [key, val] of Object.entries(assets.custom)) {
      if (typeof val !== "string" || !val.trim()) continue;
      if (!/^[a-zA-Z0-9_-]+$/.test(key)) continue;
      out[`--theme-asset-custom-${key}`] = wrap(val);
      out[`--theme-asset-custom-${key}-raw`] = val;
    }
  }
  return out;
}

/** Build `--component-<bucket>-<prop>` CSS vars from the componentStyles
 *  block. Values pass through untouched so themes can use any CSS expression. */
function componentStyleVars(
  styles: ThemeComponentStyles | undefined,
): Record<string, string> {
  if (!styles) return {};
  const out: Record<string, string> = {};
  for (const bucket of COMPONENT_BUCKETS) {
    const props = (styles as Record<string, Record<string, string> | undefined>)[bucket];
    if (!props) continue;
    for (const [prop, value] of Object.entries(props)) {
      if (typeof value !== "string" || !value.trim()) continue;
      // Same guardrail as backend — camelCase or kebab-case alnum only.
      if (!/^[a-zA-Z0-9_-]+$/.test(prop)) continue;
      out[`--component-${bucket}-${toKebab(prop)}`] = value;
    }
  }
  return out;
}

// Tracks keys we set on the previous theme so we can clear them when the
// next theme has fewer assets / component vars. Without this, switching
// from a richly-decorated theme to a plain one would leave stale vars.
let _PREV_DYNAMIC_VAR_KEYS: Set<string> = new Set();

/** ID for the injected <style> tag that carries a theme's customCSS.
 *  A single tag is reused + replaced on every theme switch. */
const CUSTOM_CSS_STYLE_ID = "hermes-theme-custom-css";

function applyCustomCSS(css: string | undefined) {
  if (typeof document === "undefined") return;
  let el = document.getElementById(CUSTOM_CSS_STYLE_ID) as HTMLStyleElement | null;
  if (!css || !css.trim()) {
    if (el) el.remove();
    return;
  }
  if (!el) {
    el = document.createElement("style");
    el.id = CUSTOM_CSS_STYLE_ID;
    el.setAttribute("data-hermes-theme-css", "true");
    document.head.appendChild(el);
  }
  el.textContent = css;
}

function applyLayoutVariant(variant: ThemeLayoutVariant | undefined) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  const final: ThemeLayoutVariant = variant ?? "standard";
  root.dataset.layoutVariant = final;
  root.style.setProperty("--theme-layout-variant", final);
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
  // Clear dynamic (asset/component) vars from the previous theme so the
  // new one starts clean — otherwise stale notched clip-paths, hero URLs,
  // etc. would bleed across theme switches.
  for (const prevKey of _PREV_DYNAMIC_VAR_KEYS) {
    root.style.removeProperty(prevKey);
  }

  const assetMap = assetVars(theme.assets);
  const componentMap = componentStyleVars(theme.componentStyles);
  _PREV_DYNAMIC_VAR_KEYS = new Set([
    ...Object.keys(assetMap),
    ...Object.keys(componentMap),
  ]);

  const vars = {
    ...paletteVars(theme.palette),
    ...typographyVars(theme.typography),
    ...layoutVars(theme.layout),
    ...overrideVars(theme.colorOverrides),
    ...assetMap,
    ...componentMap,
  };
  for (const [k, v] of Object.entries(vars)) {
    root.style.setProperty(k, v);
  }

  injectFontStylesheet(theme.typography.fontUrl);
  applyCustomCSS(theme.customCSS);
  applyLayoutVariant(theme.layoutVariant);
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
