import type { DashboardTheme, ThemeTypography, ThemeLayout } from "./types";

/**
 * Built-in dashboard themes.
 *
 * Each theme defines its own palette, typography, and layout so switching
 * themes produces visible changes beyond just color — fonts, density, and
 * corner-radius all shift to match the theme's personality.
 *
 * Theme names must stay in sync with the backend's
 * `_BUILTIN_DASHBOARD_THEMES` list in `hermes_cli/web_server.py`.
 */

// ---------------------------------------------------------------------------
// Shared typography / layout presets
// ---------------------------------------------------------------------------

/** Default system stack — neutral, safe fallback for every platform. */
const SYSTEM_SANS =
  'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const SYSTEM_MONO =
  'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace';

const DEFAULT_TYPOGRAPHY: ThemeTypography = {
  fontSans: SYSTEM_SANS,
  fontMono: SYSTEM_MONO,
  baseSize: "15px",
  lineHeight: "1.55",
  letterSpacing: "0",
};

const DEFAULT_LAYOUT: ThemeLayout = {
  radius: "0.5rem",
  density: "comfortable",
};

// ---------------------------------------------------------------------------
// Themes
// ---------------------------------------------------------------------------

export const defaultTheme: DashboardTheme = {
  name: "default",
  label: "Hermes Teal",
  description: "Classic dark teal — the canonical Hermes look",
  palette: {
    background: { hex: "#041c1c", alpha: 1 },
    midground: { hex: "#ffe6cb", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 189, 56, 0.35)",
    noiseOpacity: 1,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: DEFAULT_LAYOUT,
  terminalBackground: "#000000",
};

export const midnightTheme: DashboardTheme = {
  name: "midnight",
  label: "Midnight",
  description: "Deep blue-violet with cool accents",
  palette: {
    background: { hex: "#0a0a1f", alpha: 1 },
    midground: { hex: "#d4c8ff", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(167, 139, 250, 0.32)",
    noiseOpacity: 0.8,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Inter", ${SYSTEM_SANS}`,
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap",
    letterSpacing: "-0.005em",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.75rem",
  },
};

export const emberTheme: DashboardTheme = {
  name: "ember",
  label: "Ember",
  description: "Warm crimson and bronze — forge vibes",
  palette: {
    background: { hex: "#1a0a06", alpha: 1 },
    midground: { hex: "#ffd8b0", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(249, 115, 22, 0.38)",
    noiseOpacity: 1,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Spectral", Georgia, "Times New Roman", serif`,
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;700&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.25rem",
  },
  colorOverrides: {
    destructive: "#c92d0f",
    warning: "#f97316",
  },
};

export const monoTheme: DashboardTheme = {
  name: "mono",
  label: "Mono",
  description: "Clean grayscale — minimal and focused",
  palette: {
    background: { hex: "#0e0e0e", alpha: 1 },
    midground: { hex: "#eaeaea", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(255, 255, 255, 0.1)",
    noiseOpacity: 0.6,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"IBM Plex Sans", ${SYSTEM_SANS}`,
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0",
  },
};

export const cyberpunkTheme: DashboardTheme = {
  name: "cyberpunk",
  label: "Cyberpunk",
  description: "Neon green on black — matrix terminal",
  palette: {
    background: { hex: "#040608", alpha: 1 },
    midground: { hex: "#9bffcf", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(0, 255, 136, 0.22)",
    noiseOpacity: 1.2,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Share Tech Mono", "JetBrains Mono", ${SYSTEM_MONO}`,
    fontMono: `"Share Tech Mono", "JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;700&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0",
  },
  colorOverrides: {
    success: "#00ff88",
    warning: "#ffd700",
    destructive: "#ff0055",
  },
};

export const roseTheme: DashboardTheme = {
  name: "rose",
  label: "Rosé",
  description: "Soft pink and warm ivory — easy on the eyes",
  palette: {
    background: { hex: "#1a0f15", alpha: 1 },
    midground: { hex: "#ffd4e1", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(249, 168, 212, 0.3)",
    noiseOpacity: 0.9,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Fraunces", Georgia, serif`,
    fontMono: `"DM Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=DM+Mono:wght@400;500&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "1rem",
  },
};

/**
 * Nous Blue — the inverted "light mode" Hermes look, ported from the
 * LENS_5I overlay preset in `@nous-research/ui`.
 *
 * Unlike the other built-ins (which paint dark color directly on the
 * canvas), this theme relies on `<Backdrop />`'s foreground inversion
 * layer: an opaque white sheet at z-200 with `mix-blend-mode: difference`
 * that flips the entire stack below it. Authoring colors stay dark
 * (`#170d02` brown background, `#FFAC02` orange midground), and the
 * inversion converts them to their visual complements at paint time —
 * the orange midground reads as #0053FD Nous-blue on screen, against a
 * cream `#E8F2FD` canvas.
 *
 * Note on bg blend mode: the DS Lens uses `multiply` for LENS_5I because
 * nousnet-web's <body> is white; hermes-agent's App root is `bg-black`,
 * so we leave the bg layer's blend mode at the `difference` default —
 * `difference(#170d02, #000)` passes the bg through unchanged, and the
 * subsequent FG-difference layer then inverts it to cream. Using
 * `multiply` here would collapse the bg to pure black against the
 * `bg-black` root and produce a plain-white canvas instead of the
 * intended cream-blue.
 *
 * Source of truth for the palette: `design-language/src/ui/components/
 * overlays/lens.ts` (LENS_5I export).
 */
export const nousBlueTheme: DashboardTheme = {
  name: "nous-blue",
  label: "Nous Blue",
  description: "Light mode — vivid Nous-blue accents on cream canvas",
  palette: {
    background: { hex: "#170d02", alpha: 1 },
    midground: { hex: "#FFAC02", alpha: 1 },
    foreground: { hex: "#FFFFFF", alpha: 1 },
    // Same warm-amber as nousnet-web's overlay glow; after the FG
    // inversion it reads as a cool ultraviolet vignette in the top-left.
    warmGlow: "rgba(255, 172, 2, 0.18)",
    // Noise sits above the FG inversion and is NOT flipped, so a softer
    // multiplier keeps it from speckling over the bright post-inversion
    // canvas.
    noiseOpacity: 0.4,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: DEFAULT_LAYOUT,
  // Inverted page: the embedded terminal is below the FG layer too, so
  // a `#000000` source paints as visual white — i.e. a proper light-mode
  // terminal pane. xterm picks lighter palette colors against the "black"
  // canvas, which then read as dark text on screen post-inversion.
  terminalBackground: "#000000",
  componentStyles: {
    backdrop: {
      // Lower than LENS_5I.Lens.fillerOpacity (0.06). The filler texture
      // gets amplified post-inversion: small variations against the deep
      // `#170d02` source bg are barely visible, but those same variations
      // against the bright `#E8F2FD` post-inversion canvas read as a
      // heavy cloud/marble pattern — especially on near-empty pages
      // (loading spinners, blank states). 0.02 keeps subtle grain
      // without overwhelming the canvas.
      fillerOpacity: "0.02",
    },
  },
  // Pre-invert absolute-hex tokens so they read as their familiar colors
  // through the FG difference layer. e.g. source #04D3C9 (cyan) is what
  // gets painted, and `255 - channel` flips it to #FB2C36 (red) on screen.
  // Without these, the default destructive/success/warning tokens would
  // appear as their unintuitive complements.
  colorOverrides: {
    destructive: "#04d3c9",
    destructiveForeground: "#000000",
    success: "#b5217f",
    warning: "#0042c7",
  },
  // Pre-inverted data-series accents for the Analytics/Models token
  // charts. The defaults (#ffe6cb cream + #34d399 emerald) would render
  // through the FG difference layer as dark navy + hot-coral on the
  // bright Nous-blue canvas — the coral is the "red" users see for
  // Output values without these overrides. Source → on-screen:
  //   Input:  #ffe6cb → #001934 (dark navy)        ← unchanged
  //   Output: #ffac02 → #0053fd (vivid Nous-blue)  ← brand accent
  // Input keeps the cream source so it stays a neutral, low-contrast
  // dark-blue against the cream canvas; output paints as the brand
  // Nous-blue so the "primary" series in token-flow charts reads as
  // the highlight color, matching the rest of the inverted UI chrome.
  seriesColors: {
    inputTokenAccent: "#ffe6cb",
    outputTokenAccent: "#ffac02",
  },
  // Explicit picker swatch — the raw palette hex (`#170d02`, `#FFAC02`,
  // amber rgba) doesn't reflect what users see after the FG inversion,
  // so we paint the post-inversion visual triplet directly:
  //   white → vivid Nous-blue → cream/light-blue
  // matching the actual on-screen rendering of the theme.
  swatchColors: ["#FFFFFF", "#0053FD", "#E8F2FD"],
};

/**
 * JARVIS · Glassify — the signature brand theme.
 *
 * A dark, modern-luxury reskin: smoked near-black glass canvas, a warm
 * platinum/champagne midground, and a gold accent that drives primary
 * actions and focus rings. The "glass" comes from two cooperating layers:
 *
 *   1. `colorOverrides` make every elevated surface (card, popover,
 *      secondary, muted, accent) *translucent* — they're authored as
 *      `color-mix(... transparent)` instead of opaque so what's behind
 *      them shows through.
 *   2. `customCSS` adds the frost: `backdrop-filter: blur()` on those
 *      same surfaces plus the header/sidebar chrome, with a soft inner
 *      highlight + deep drop shadow so panels read as lifted glass.
 *
 * Larger `radius` (1rem) keeps panel corners soft, and the low
 * `noiseOpacity` keeps the canvas sleek rather than gritty.
 */
export const glassifyTheme: DashboardTheme = {
  name: "glassify",
  label: "JARVIS Glass",
  description: "Dark modern luxury — frosted glass surfaces and champagne accents",
  palette: {
    background: { hex: "#0a0b0f", alpha: 1 },
    midground: { hex: "#e8e2d4", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(201, 168, 106, 0.22)",
    noiseOpacity: 0.35,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Inter", ${SYSTEM_SANS}`,
    fontDisplay: `"Manrope", "Inter", ${SYSTEM_SANS}`,
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap",
    letterSpacing: "-0.01em",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "1rem",
  },
  terminalBackground: "#0a0b0f",
  colorOverrides: {
    // Translucent so the backdrop-filter blur below actually has something
    // to frost. Authored against the base hex + transparent rather than
    // the opaque DS cascade defaults.
    card: "color-mix(in srgb, var(--background-base) 52%, transparent)",
    popover: "color-mix(in srgb, var(--background-base) 82%, transparent)",
    secondary: "color-mix(in srgb, var(--midground-base) 8%, transparent)",
    muted: "color-mix(in srgb, var(--midground-base) 10%, transparent)",
    accent: "color-mix(in srgb, #c9a86a 18%, transparent)",
    border: "color-mix(in srgb, var(--midground-base) 13%, transparent)",
    input: "color-mix(in srgb, var(--midground-base) 13%, transparent)",
    // Champagne gold accent for primary actions + focus rings.
    primary: "#c9a86a",
    primaryForeground: "#0a0b0f",
    ring: "#c9a86a",
    warning: "#e0b352",
  },
  seriesColors: {
    inputTokenAccent: "#e8e2d4",
    outputTokenAccent: "#c9a86a",
  },
  swatchColors: ["#0a0b0f", "#c9a86a", "#e8e2d4"],
  customCSS: `
/* ────────────────────────────────────────────────────────────────
   JARVIS · Glassify — frosted glass chrome + champagne luxe accents
   ──────────────────────────────────────────────────────────────── */

/* Frost every elevated surface. Translucency comes from colorOverrides;
   this adds the blur, a lifted drop shadow, and a hairline top highlight
   so panels read as a pane of smoked glass rather than a flat fill. */
.bg-card,
.bg-popover,
.bg-secondary,
.bg-muted {
  backdrop-filter: blur(18px) saturate(150%);
  -webkit-backdrop-filter: blur(18px) saturate(150%);
  box-shadow:
    0 12px 40px -14px rgba(0, 0, 0, 0.7),
    inset 0 1px 0 color-mix(in srgb, var(--midground-base) 14%, transparent);
}

/* Header + sidebar nav float over scrolling content with a deeper frost. */
header,
aside {
  backdrop-filter: blur(24px) saturate(140%);
  -webkit-backdrop-filter: blur(24px) saturate(140%);
}

/* Champagne sheen on the primary action so gold buttons read as polished
   metal rather than a flat swatch. */
.bg-primary {
  background-image: linear-gradient(
    180deg,
    color-mix(in srgb, #ffffff 22%, transparent) 0%,
    transparent 55%
  );
}
`,
};

/**
 * Same look as ``defaultTheme`` but with a larger root font size, looser
 * line-height, and ``spacious`` density so every rem-based size in the
 * dashboard scales up. For users who find the default 15px UI too dense.
 */
export const defaultLargeTheme: DashboardTheme = {
  name: "default-large",
  label: "Hermes Teal (Large)",
  description: "Hermes Teal with bigger fonts and roomier spacing",
  palette: defaultTheme.palette,
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    baseSize: "18px",
    lineHeight: "1.65",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    density: "spacious",
  },
};

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  glassify: glassifyTheme,
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  "nous-blue": nousBlueTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
};
