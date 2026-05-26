/**
 * Dashboard theme model.
 *
 * Themes customise three orthogonal layers:
 *
 *   1. `palette`       — the 3-layer color triplet (background/midground/
 *                         foreground) + warm-glow + noise opacity. The
 *                         design-system cascade in `src/index.css` derives
 *                         every shadcn-compat token (card, muted, border,
 *                         primary, etc.) from this triplet via `color-mix()`.
 *   2. `typography`    — font families, base font size, line height,
 *                         letter spacing. An optional `fontUrl` is injected
 *                         as `<link rel="stylesheet">` so self-hosted and
 *                         Google/Bunny/etc-hosted fonts both work.
 *   3. `layout`        — corner radius and density (spacing multiplier).
 *
 * Plus an optional `colorOverrides` escape hatch for themes that want to
 * pin specific shadcn tokens to exact values (e.g. a pastel theme that
 * needs a softer `destructive` red than the derived default).
 */

/** A color layer: hex base + alpha (0–1). */
export interface ThemeLayer {
  alpha: number;
  hex: string;
}

export interface ThemePalette {
  /** Deepest canvas color (typically near-black). */
  background: ThemeLayer;
  /** Primary text + accent. Most UI chrome reads this. */
  midground: ThemeLayer;
  /** Top-layer highlight. In LENS_0 this is white @ alpha 0 — invisible by
   *  default but still drives `--color-ring`-style accents. */
  foreground: ThemeLayer;
  /** Warm vignette color for <Backdrop />, as an rgba() string. */
  warmGlow: string;
  /** Scalar multiplier (0–1.2) on the noise overlay. Lower for softer themes
   *  like Mono and Rosé, higher for grittier themes like Cyberpunk. */
  noiseOpacity: number;
}

export interface ThemeTypography {
  /** CSS font-family stack for sans-serif body copy. */
  fontSans: string;
  /** CSS font-family stack for monospace / code blocks. */
  fontMono: string;
  /** Optional display/heading font stack. Falls back to `fontSans`. */
  fontDisplay?: string;
  /** Optional external stylesheet URL (e.g. Google Fonts, Bunny Fonts,
   *  self-hosted .woff2 @font-face sheet). Injected as a <link> in <head>
   *  on theme switch. Same URL is never injected twice. */
  fontUrl?: string;
  /** Root font size (controls rem scale). Example: `"14px"`, `"16px"`. */
  baseSize: string;
  /** Default line-height. Example: `"1.5"`, `"1.65"`. */
  lineHeight: string;
  /** Default letter-spacing. Example: `"0"`, `"0.01em"`, `"-0.01em"`. */
  letterSpacing: string;
}

export type ThemeDensity = "compact" | "comfortable" | "spacious";

export interface ThemeLayout {
  /** Corner-radius token. Example: `"0"`, `"0.25rem"`, `"0.5rem"`,
   *  `"1rem"`. Maps to `--radius` and cascades into every component. */
  radius: string;
  /** Spacing multiplier. `compact` = 0.85, `comfortable` = 1.0 (default),
   *  `spacious` = 1.2. Applied via the `--spacing-mul` CSS var. */
  density: ThemeDensity;
}

/** Overall layout variant the shell renders. `standard` = default single-
 *  column page layout. `cockpit` = reserves a left sidebar rail for a
 *  plugin slot (intended for HUD-style themes with persistent status panels).
 *  `tiled` = relaxes the main content max-width so pages can use the full
 *  viewport width. Themes set this; plugins react via CSS vars /
 *  `[data-layout-variant="..."]` selectors. */
export type ThemeLayoutVariant = "standard" | "cockpit" | "tiled";

/** Named hero/background assets a theme can populate. Each value is
 *  emitted as a CSS var (`--theme-asset-<name>`). The default shell
 *  consumes `bg` in `<Backdrop />` when present; other slots are
 *  plugin-facing — a cockpit sidebar plugin reads `--theme-asset-hero`
 *  to render its hero render without coupling to the theme name. */
export interface ThemeAssets {
  /** Full-viewport background image URL, injected under the noise layer. */
  bg?: string;
  /** Hero render (Gundam, mascot, wallpaper) — for plugin sidebars/overlays. */
  hero?: string;
  /** Logo mark — header slot consumers use this. */
  logo?: string;
  /** Faction/brand crest — header-left decoration. */
  crest?: string;
  /** Secondary sidebar illustration. */
  sidebar?: string;
  /** Alternate header artwork. */
  header?: string;
  /** User-defined named assets. Keyed by [a-zA-Z0-9_-] only.
   *  Emitted as `--theme-asset-custom-<key>`. */
  custom?: Record<string, string>;
}

/** Component-style override buckets. Each bucket's entries become CSS
 *  vars (`--component-<bucket>-<kebab-property>`) that shell components
 *  (Card, Backdrop, App header/footer, etc.) read. Values are plain CSS
 *  strings — we don't parse them, so themes can use `clip-path`,
 *  `border-image`, `background`, `box-shadow`, and anything else CSS
 *  accepts. */
export interface ThemeComponentStyles {
  card?: Record<string, string>;
  header?: Record<string, string>;
  footer?: Record<string, string>;
  sidebar?: Record<string, string>;
  tab?: Record<string, string>;
  progress?: Record<string, string>;
  badge?: Record<string, string>;
  backdrop?: Record<string, string>;
  page?: Record<string, string>;
}

/** Optional hex overrides keyed by shadcn-compat token name (without the
 *  `--color-` prefix). Any key set here wins over the DS cascade. */
export interface ThemeColorOverrides {
  card?: string;
  cardForeground?: string;
  popover?: string;
  popoverForeground?: string;
  primary?: string;
  primaryForeground?: string;
  secondary?: string;
  secondaryForeground?: string;
  muted?: string;
  mutedForeground?: string;
  accent?: string;
  accentForeground?: string;
  destructive?: string;
  destructiveForeground?: string;
  success?: string;
  warning?: string;
  border?: string;
  input?: string;
  ring?: string;
}

export interface DashboardTheme {
  description: string;
  label: string;
  name: string;
  palette: ThemePalette;
  typography: ThemeTypography;
  layout: ThemeLayout;
  /** Overall shell layout. Defaults to `"standard"` when absent. */
  layoutVariant?: ThemeLayoutVariant;
  /** Named + custom asset URLs exposed as CSS vars on theme apply. */
  assets?: ThemeAssets;
  /** Raw CSS injected as a scoped `<style>` tag on theme apply, cleaned up
   *  on theme switch. Intended for selector-level chrome that's too
   *  expressive for componentStyles alone (e.g. `::before` pseudo-elements,
   *  complex animations, media queries). */
  customCSS?: string;
  /** Per-component CSS-var overrides. See `ThemeComponentStyles`. */
  componentStyles?: ThemeComponentStyles;
  colorOverrides?: ThemeColorOverrides;
}

/**
 * Wire response shape for `GET /api/dashboard/themes`.
 *
 * The `themes` list is intentionally partial — built-in themes are fully
 * defined in `presets.ts`; user themes carry their full definition so the
 * client can apply them without a second round-trip.
 */
export interface ThemeListEntry {
  description: string;
  label: string;
  name: string;
  /** Full theme definition. Present for user-defined themes loaded from
   *  `~/.hermes/dashboard-themes/*.yaml`; undefined for built-ins (the
   *  client already has those in `BUILTIN_THEMES`). */
  definition?: DashboardTheme;
}

export interface ThemeListResponse {
  active: string;
  themes: ThemeListEntry[];
}
