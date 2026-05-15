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
 * Mission Control — flat dark dashboard styled to match the Mission
 * Control project (github.com/builderz-labs/mission-control) as closely
 * as Hermes's theme system allows.
 *
 * Compared with the other Hermes presets this one is *aggressively
 * de-Hermes-ified*:
 *
 *   - midground is a near-white (#e6ebf0), not the accent — Hermes's
 *     DS cascade derives `--color-foreground` from midground, so body
 *     text needs to be readable greyscale. The cyan accent is bolted
 *     on via `colorOverrides` (primary/ring/accent/secondary).
 *   - All decorative chrome that ships in the default Hermes look is
 *     killed via `componentStyles` overrides: the diagonal clip-path
 *     borders on the sidebar/header/cards/tabs, the warm-glow vignette
 *     (set warmGlow: "transparent"), the SVG noise overlay
 *     (noiseOpacity: 0), and the filler-bg jpeg in `<Backdrop>` (set
 *     --component-backdrop-filler-opacity: 0).
 *   - `customCSS` retargets `--font-mondwest` from the bundled
 *     decorative display face to plain Inter, so sidebar nav items and
 *     section labels drop the retro-display vibe.
 *
 * Palette borrowed verbatim from MC's globals.css `.dark` block.
 */
/**
 * Mission Control — palette ported verbatim from
 * github.com/builderz-labs/mission-control's `globals.css` `.dark` block,
 * with the HSL values mechanically converted to hex so the Hermes theme
 * system (which expects hex + alpha) can apply them.
 *
 * Mapping of MC's tokens to Hermes's three-layer model:
 *
 *   MC --background  (215 27% 4%)  → palette.background.hex  = #080a0e
 *   MC --foreground  (210 20% 92%) → palette.midground.hex   = #e3e9ef
 *      (Hermes drives the global `color: var(--midground)` rule off
 *       this — so it MUST be a readable greyscale, not the accent,
 *       or every label across the dashboard turns cyan.)
 *   palette.foreground stays at white+α0 (Hermes convention: the
 *       foreground layer is an invisible overlay slot, NOT the text).
 *
 * The remaining MC tokens are pinned exactly via `colorOverrides`. The
 * Hermes DS cascade would otherwise derive surface stops as 4%/8%/15%
 * midground mixes, which on a flat-dark canvas comes out almost
 * indistinguishable from the background — MC's hand-tuned surface
 * hierarchy reads much better.
 *
 * The decorative chrome (clip-path notches on sidebar/header/card,
 * warm-glow vignette, SVG noise grain, filler-bg jpeg) is killed via
 * `componentStyles` + `warmGlow: "transparent"` + `noiseOpacity: 0` —
 * MC is a flat-design dashboard with crisp 1px borders and no overlay
 * effects.
 *
 * Inter (sans) + JetBrains Mono (mono) match MC's Next.js stack; the
 * font file is pulled from Google Fonts on first apply.
 */
export const missionControlTheme: DashboardTheme = {
  name: "mission-control",
  label: "Mission Control",
  description: "Flat dark — palette and chrome copied from Mission Control",
  palette: {
    background: { hex: "#080a0e", alpha: 1 },
    midground: { hex: "#e3e9ef", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "transparent",
    noiseOpacity: 0,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Inter", ${SYSTEM_SANS}`,
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontDisplay: `"Inter", ${SYSTEM_SANS}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap",
    baseSize: "14px",
    lineHeight: "1.5",
    letterSpacing: "-0.005em",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.5rem",
    density: "comfortable",
  },
  componentStyles: {
    // Sidebar + header are flat dark panels with a single border-bottom;
    // none of Hermes's clip-path / border-image flourishes.
    sidebar: {
      background: "#0a0d12",
      clipPath: "none",
      borderImage: "none",
    },
    header: {
      background: "#0a0d12",
      clipPath: "none",
      borderImage: "none",
    },
    // Cards: explicit fill matching MC's --card; no shadow, no clip.
    card: {
      background: "#0e1219",
      clipPath: "none",
      borderImage: "none",
      boxShadow: "none",
    },
    tab: {
      clipPath: "none",
    },
    // Kill the filler-bg jpeg layer (z-2 in <Backdrop />); combined with
    // warmGlow=transparent and noiseOpacity=0 the canvas collapses to a
    // single flat --background-base fill.
    backdrop: {
      fillerOpacity: "0",
    },
  },
  // Aggressive global overrides — Hermes's Tailwind config generates
  // every `bg-card` / `bg-muted` / `bg-popover` / etc. class with TWO
  // rules: a fallback `background-color: var(--midground-base)` and a
  // preferred `color-mix(srgb, midground-base N%, background-base)`.
  // When midground is a light text-grey (which it MUST be in a dark
  // theme — body text reads off `color: var(--midground)`), the
  // fallback resolves to LIGHT and any surface that ends up using it
  // (e.g. older WebKits, or any cascade quirk) renders near-white.
  //
  // Rather than rely on color-mix always winning, we paint every
  // semantic surface to an explicit hex so a flat-dark look is
  // guaranteed regardless of how Tailwind resolves the cascade.
  customCSS: `
    :root {
      --font-mondwest: "Inter", system-ui, sans-serif;
    }
    /* Semantic surfaces — pinned to MC's palette explicitly. */
    .bg-card, .bg-card\\/50, .bg-card\\/80, .bg-card\\/95 {
      background-color: #0e1219 !important;
    }
    .bg-muted { background-color: #1c212b !important; }
    .bg-muted\\/10 { background-color: rgba(28, 33, 43, 0.10) !important; }
    .bg-muted\\/20 { background-color: rgba(28, 33, 43, 0.30) !important; }
    .bg-muted\\/30 { background-color: rgba(28, 33, 43, 0.40) !important; }
    .bg-muted\\/40 { background-color: rgba(28, 33, 43, 0.50) !important; }
    .bg-muted\\/50 { background-color: rgba(28, 33, 43, 0.65) !important; }
    .bg-muted\\/60 { background-color: rgba(28, 33, 43, 0.75) !important; }
    .bg-popover, .bg-popover\\/95 { background-color: #0e1219 !important; }
    .bg-accent { background-color: #1c212b !important; }
    .bg-accent\\/50 { background-color: rgba(28, 33, 43, 0.5) !important; }
    .bg-secondary { background-color: #161c25 !important; }
    .bg-primary { background-color: #28d2ef !important; }
    .bg-background, .bg-background\\/85, .bg-background\\/95 {
      background-color: #080a0e !important;
    }
    .bg-background-base, .bg-background-base\\/95 {
      background-color: rgba(8, 10, 14, 0.95) !important;
    }

    /* Borders — same problem: cascade derives them from midground at
       15% over transparent → faint light line on dark bg, fights MC's
       crisp 1px-border aesthetic. Pin to muted-grey directly. */
    .border-border { border-color: #1c212b !important; }
    .border-current\\/10 { border-color: rgba(227, 233, 239, 0.10) !important; }
    .border-current\\/20 { border-color: rgba(227, 233, 239, 0.20) !important; }

    /* Sidebar inactive nav items use opacity-60 on inherited body color
       — readable but quite dim against a flat-dark sidebar. Lift them
       a notch so menu items are easy to scan. Active state keeps full
       brightness via the existing text-midground class. */
    aside nav a[href].opacity-60 {
      opacity: 0.82;
    }
    aside nav a[href].opacity-60:hover {
      opacity: 1;
    }

    /* Make sure muted-foreground (column subtitles, "5M AGO" badges,
       hint text) stays visible on the flat-dark canvas. */
    .text-muted-foreground, .text-muted-foreground\\/70 {
      color: #94a3b8 !important;
    }

    /* Primary text — the body color rule (color: var(--midground))
       already paints text in #e3e9ef, but a handful of places still
       use the raw --color-foreground token. Force them aligned. */
    .text-foreground, .text-foreground\\/80, .text-foreground\\/90 {
      color: #e3e9ef !important;
    }
  `,
  colorOverrides: {
    // MC --primary / --ring / --void-cyan (187 82% 53%).
    ring: "#28d2ef",
    primary: "#28d2ef",
    primaryForeground: "#0a0e14",
    // MC --accent (220 20% 14%) + --accent-foreground (210 20% 92%).
    accent: "#1c212b",
    accentForeground: "#e3e9ef",
    // MC --secondary (220 25% 11%) + --secondary-foreground (210 20% 92%).
    secondary: "#161c25",
    secondaryForeground: "#e3e9ef",
    // MC --card (220 30% 8%) + --card-foreground (210 20% 92%).
    card: "#0e1219",
    cardForeground: "#e3e9ef",
    // MC --popover same as card.
    popover: "#0e1219",
    popoverForeground: "#e3e9ef",
    // MC --muted (220 20% 14%) + --muted-foreground (220 15% 50%).
    muted: "#1c212b",
    mutedForeground: "#717d8e",
    // MC --border / --input (220 20% 14%).
    border: "#1c212b",
    input: "#1c212b",
    // MC --destructive (0 72% 51%).
    destructive: "#dc2828",
    destructiveForeground: "#fafafa",
    // MC --success (160 60% 52%) + --warning (38 92% 50%).
    success: "#37c298",
    warning: "#f59e0b",
  },
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
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
  "mission-control": missionControlTheme,
};
