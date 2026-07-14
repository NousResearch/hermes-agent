import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { BUILTIN_THEMES, defaultTheme } from "./presets";
import {
  FONT_CHOICES,
  THEME_DEFAULT_FONT_ID,
  getFontChoice,
  type FontChoice,
} from "./fonts";
import type {
  DashboardTheme,
  ThemeAssets,
  ThemeColorOverrides,
  ThemeComponentStyles,
  ThemeDensity,
  ThemeLayer,
  ThemeLayout,
  ThemeLayoutVariant,
  ThemeListEntry,
  ThemePalette,
  ThemeSeriesColors,
  ThemeTypography,
} from "./types";
import { api } from "@/lib/api";

/** LocalStorage key holding the active theme NAME (built-in id or user YAML
 *  name). Pre-applied at module load (see `bootstrapThemeFromCache`) before the
 *  React tree mounts to avoid a visible flash of the default palette on
 *  theme-overridden installs. */
const STORAGE_KEY = "hermes-dashboard-theme";

/** LocalStorage key holding the full user-theme DEFINITIONS map
 *  (name → DashboardTheme) cached from `/api/dashboard/themes`. The name alone
 *  isn't enough to repaint a custom theme on reload: built-ins ship in the
 *  bundle, but user YAMLs only exist server-side, so without this cache
 *  `resolveTheme(name)` falls back to `defaultTheme` until the API responds —
 *  the GUNNY-COMMAND-style first-load flash. Seeding from this cache makes the
 *  first render resolve the real definition. */
const THEME_DEFS_STORAGE_KEY = "hermes-dashboard-theme-defs";

/** LocalStorage key for the font override (independent of theme). Holds a
 *  font id from the catalog in `fonts.ts`, or the `THEME_DEFAULT_FONT_ID`
 *  sentinel / absent = "use the active theme's font". Pre-applied before
 *  the React tree mounts (see `main.tsx`) to avoid a font flash. */
const FONT_STORAGE_KEY = "hermes-dashboard-font";

/** Renames of built-in theme keys we've shipped previously. Without this,
 *  users who saved one of the old names in localStorage (or had it
 *  persisted server-side) would silently fall back to `defaultTheme`
 *  because the lookup in `resolveTheme` no longer finds the stale key.
 *  Keep entries here until enough release cycles have passed that we can
 *  reasonably assume nobody still has the old value persisted. */
const THEME_NAME_ALIASES: Record<string, string> = {
  // Renamed during the LENS_5I port + Nous-blue rebrand.
  "lens-5i": "nous-blue",
};

function migrateThemeName(name: string): string {
  return THEME_NAME_ALIASES[name] ?? name;
}

function readStoredThemeName(): string {
  if (typeof window === "undefined") return "default";
  try {
    return migrateThemeName(window.localStorage.getItem(STORAGE_KEY) ?? "default");
  } catch {
    return "default";
  }
}

function readCachedThemeDefs(): Record<string, DashboardTheme> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(THEME_DEFS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object"
      ? (parsed as Record<string, DashboardTheme>)
      : {};
  } catch {
    return {};
  }
}

function writeCachedThemeDefs(defs: Record<string, DashboardTheme>): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(THEME_DEFS_STORAGE_KEY, JSON.stringify(defs));
  } catch {
    // Quota exceeded / private mode — the cache is best-effort; without it we
    // simply fall back to the (slower) API-resolved path on next load.
  }
}

/** Server-injected active-theme bootstrap. The web server writes
 *  `window.__HERMES_THEME_BOOTSTRAP__ = {name, definition}` into index.html
 *  (an inline classic <script>, so it runs before the deferred module bundle).
 *  This is the ONLY source available on a first-ever load with an empty
 *  localStorage cache, so it's consulted first — it kills the default-palette
 *  flash even before the API responds and before any cache exists. `definition`
 *  is null for built-in active themes (they ship in the bundle) and the full
 *  normalised theme object for user YAMLs. */
interface ThemeBootstrap {
  name: string;
  definition: DashboardTheme | null;
}

function readServerThemeBootstrap(): ThemeBootstrap | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = (window as unknown as {
      __HERMES_THEME_BOOTSTRAP__?: unknown;
    }).__HERMES_THEME_BOOTSTRAP__;
    if (!raw || typeof raw !== "object") return null;
    const obj = raw as { name?: unknown; definition?: unknown };
    if (typeof obj.name !== "string" || !obj.name) return null;
    return {
      name: migrateThemeName(obj.name),
      definition:
        obj.definition && typeof obj.definition === "object"
          ? (obj.definition as DashboardTheme)
          : null,
    };
  } catch {
    return null;
  }
}

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

/** Map data-series accents to their CSS vars. Themes omit either field to
 *  inherit the `:root` default from `index.css`; when omitted we also
 *  proactively clear any leftover value from a previous theme so switches
 *  don't carry stale colors. */
const SERIES_KEY_TO_VAR: Record<keyof ThemeSeriesColors, string> = {
  inputTokenAccent: "--series-input-token",
  outputTokenAccent: "--series-output-token",
};

const ALL_SERIES_VARS = Object.values(SERIES_KEY_TO_VAR);

function seriesColorVars(
  series: ThemeSeriesColors | undefined,
): Record<string, string> {
  if (!series) return {};
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(series)) {
    if (!value) continue;
    const cssVar = SERIES_KEY_TO_VAR[key as keyof ThemeSeriesColors];
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
// Font override (independent of theme)
// ---------------------------------------------------------------------------

/** The active font-override id, mirrored at module scope so `applyTheme`
 *  can re-assert it after every theme switch (theme application rewrites
 *  `--theme-font-sans`, so the override has to win again afterwards). */
let _ACTIVE_FONT_OVERRIDE: string = THEME_DEFAULT_FONT_ID;

/** Apply (or clear) the font override on `:root`. When a catalog font is
 *  active we override `--theme-font-sans` and `--theme-font-display` and
 *  inject its webfont; the theme keeps ownership of `--theme-font-mono`
 *  (code/terminal) so picking a body font doesn't mangle code blocks.
 *  Passing the theme-default sentinel removes the override so the theme's
 *  own font shows through. */
function applyFontOverride(fontId: string | undefined) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  const choice: FontChoice | undefined = getFontChoice(fontId);
  if (!choice) {
    // Clear → fall back to whatever the active theme set (applyTheme already
    // wrote the theme's --theme-font-sans/-display before this runs).
    root.style.removeProperty("--theme-font-override-sans");
    return;
  }
  injectFontStylesheet(choice.fontUrl);
  // Set both the override marker var (used by the picker for diagnostics)
  // and the live consumed vars. We re-set the consumed vars directly so the
  // change is immediate and survives the next applyTheme via _ACTIVE_FONT_OVERRIDE.
  root.style.setProperty("--theme-font-override-sans", choice.stack);
  root.style.setProperty("--theme-font-sans", choice.stack);
  root.style.setProperty("--theme-font-display", choice.stack);
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
  // Same clear-then-set for series colors so a theme that defines them
  // (e.g. Nous Blue) doesn't leave its values behind when the user
  // switches to a theme that inherits the `:root` defaults.
  for (const cssVar of ALL_SERIES_VARS) {
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
    ...seriesColorVars(theme.seriesColors),
    ...assetMap,
    ...componentMap,
  };
  for (const [k, v] of Object.entries(vars)) {
    root.style.setProperty(k, v);
  }

  injectFontStylesheet(theme.typography.fontUrl);
  applyCustomCSS(theme.customCSS);
  applyLayoutVariant(theme.layoutVariant);

  // Terminal colors — read by ChatPage via useTheme(); also available as CSS vars.
  root.style.setProperty(
    "--theme-terminal-background",
    theme.terminalBackground ?? "#000000",
  );
  root.style.setProperty(
    "--theme-terminal-foreground",
    theme.terminalForeground ?? "#f0e6d2",
  );

  // Re-assert the font override last: theme application just rewrote
  // --theme-font-sans/-display, so an active override has to win again.
  applyFontOverride(_ACTIVE_FONT_OVERRIDE);
}

/** Pre-apply the active theme's CSS vars synchronously at module load — before
 *  React mounts and the browser paints. Resolution order:
 *    1. Server bootstrap (`window.__HERMES_THEME_BOOTSTRAP__`) — authoritative
 *       and present even on a first-ever load with no localStorage cache.
 *    2. localStorage stored name → bundled built-in, or cached user-theme def.
 *  The server bootstrap wins so a custom active theme (e.g. GUNNY-COMMAND)
 *  paints correctly on the very first visit, and so a stale localStorage name
 *  can't override an active theme the server has since changed. When nothing
 *  resolves we apply nothing and let the normal React flow paint default, so we
 *  never paint default twice. */
function bootstrapThemeFromCache(): void {
  if (typeof document === "undefined") return;

  const server = readServerThemeBootstrap();
  if (server) {
    const serverTheme =
      BUILTIN_THEMES[server.name] ?? server.definition ?? undefined;
    if (serverTheme) {
      applyTheme(serverTheme);
      return;
    }
  }

  const name = readStoredThemeName();
  const theme = BUILTIN_THEMES[name] ?? readCachedThemeDefs()[name];
  if (theme) applyTheme(theme);
}

bootstrapThemeFromCache();

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function ThemeProvider({ children }: { children: ReactNode }) {
  /** Name of the currently active theme (built-in id or user YAML name).
   *  Server bootstrap wins over localStorage so a first-ever load (no stored
   *  name) and a stale stored name both resolve to the server's active theme. */
  const [themeName, setThemeName] = useState<string>(() => {
    const server = readServerThemeBootstrap();
    if (server) return server.name;
    if (typeof window === "undefined") return "default";
    const stored = window.localStorage.getItem(STORAGE_KEY) ?? "default";
    const migrated = migrateThemeName(stored);
    // Write the migrated name back so future reads converge on the new
    // key and we eventually retire the alias entry.
    if (migrated !== stored) {
      window.localStorage.setItem(STORAGE_KEY, migrated);
    }
    return migrated;
  });

  /** All selectable themes (shown in the picker). Starts with just the
   *  built-ins; the API call below merges in user themes. */
  const [availableThemes, setAvailableThemes] = useState<ThemeListEntry[]>(() =>
    Object.values(BUILTIN_THEMES).map((t) => ({
      name: t.name,
      label: t.label,
      description: t.description,
    })),
  );

  /** Full definitions for user themes keyed by name — the API provides
   *  these so custom YAMLs apply without a client-side stub. Seeded from the
   *  localStorage cache so a custom active theme resolves on the FIRST render
   *  (no default-palette flash) instead of only after the API responds. */
  const [userThemeDefs, setUserThemeDefs] = useState<
    Record<string, DashboardTheme>
  >(() => {
    const cached = readCachedThemeDefs();
    // Fold in the server-bootstrapped definition so the active user theme
    // resolves on the first render even when localStorage has no cache yet.
    const server = readServerThemeBootstrap();
    if (server?.definition) {
      return { ...cached, [server.name]: server.definition };
    }
    return cached;
  });

  /** Active font-override id (independent of theme). `THEME_DEFAULT_FONT_ID`
   *  = no override. Seeded from localStorage so it's applied flash-free. */
  const [fontId, setFontId] = useState<string>(() => {
    if (typeof window === "undefined") return THEME_DEFAULT_FONT_ID;
    const stored = window.localStorage.getItem(FONT_STORAGE_KEY);
    const valid = stored && getFontChoice(stored) ? stored : THEME_DEFAULT_FONT_ID;
    _ACTIVE_FONT_OVERRIDE = valid;
    return valid;
  });

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

  // Apply the active theme (and re-assert the font override at its tail)
  // whenever the theme, the resolver, OR the font override changes. Folding
  // font into the same layout effect means clearing the override re-runs
  // applyTheme, which restores the theme's own font; setting it re-asserts the
  // override. useLayoutEffect (not useEffect) commits CSS vars before paint —
  // the module-load bootstrap covers the very first paint, and this keeps later
  // theme/font switches flash-free too.
  useLayoutEffect(() => {
    _ACTIVE_FONT_OVERRIDE = fontId;
    applyTheme(resolveTheme(themeName));
  }, [themeName, resolveTheme, fontId]);

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
              definition: t.definition,
            })),
          );
          // Index any definitions the server shipped (user themes).
          const defs: Record<string, DashboardTheme> = {};
          for (const entry of resp.themes) {
            if (entry.definition) {
              defs[entry.name] = entry.definition;
            }
          }
          if (Object.keys(defs).length > 0) {
            setUserThemeDefs(defs);
            // Cache the freshest defs so the next reload can pre-apply a custom
            // active theme before first paint instead of flashing default.
            writeCachedThemeDefs(defs);
          }
        }
        if (resp.active) {
          const migratedActive = migrateThemeName(resp.active);
          if (migratedActive !== themeName) {
            setThemeName(migratedActive);
            window.localStorage.setItem(STORAGE_KEY, migratedActive);
          }
          // If the server is still persisting the stale key, push the
          // migrated value back so it converges too — otherwise every
          // future page load would re-trigger this branch.
          if (migratedActive !== resp.active) {
            api.setTheme(migratedActive).catch(() => {});
          }
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load the server-persisted font override once on mount. The server is
  // the source of truth across browsers; localStorage just avoids the flash.
  useEffect(() => {
    let cancelled = false;
    api
      .getFontPref()
      .then((resp) => {
        if (cancelled) return;
        const serverId =
          resp?.font && getFontChoice(resp.font) ? resp.font : THEME_DEFAULT_FONT_ID;
        if (serverId !== fontId) {
          setFontId(serverId);
          if (typeof window !== "undefined") {
            window.localStorage.setItem(FONT_STORAGE_KEY, serverId);
          }
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

  const setFont = useCallback((id: string) => {
    const next = getFontChoice(id) ? id : THEME_DEFAULT_FONT_ID;
    setFontId(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(FONT_STORAGE_KEY, next);
    }
    api.setFontPref(next).catch(() => {});
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({
      theme: resolveTheme(themeName),
      themeName,
      availableThemes,
      setTheme,
      fontId,
      fontChoices: FONT_CHOICES,
      setFont,
    }),
    [themeName, availableThemes, setTheme, resolveTheme, fontId, setFont],
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
  fontId: THEME_DEFAULT_FONT_ID,
  fontChoices: FONT_CHOICES,
  setFont: () => {},
});

interface ThemeContextValue {
  availableThemes: ThemeListEntry[];
  setTheme: (name: string) => void;
  theme: DashboardTheme;
  themeName: string;
  /** Active font-override id (`THEME_DEFAULT_FONT_ID` = no override). */
  fontId: string;
  /** Curated font catalog for the picker. */
  fontChoices: FontChoice[];
  /** Set the font override (independent of theme). */
  setFont: (id: string) => void;
}
