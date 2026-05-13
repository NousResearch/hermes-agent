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

export const haoyuanlinTheme: DashboardTheme = {
  name: "haoyuanlin",
  label: "Haoyuan Brutalist",
  description: "Memphis x Brutalist cream canvas, hard ink borders, and punchy accents",
  palette: {
    background: { hex: "#FAF8F5", alpha: 1 },
    midground: { hex: "#1A1A1A", alpha: 1 },
    foreground: { hex: "#FFD600", alpha: 1 },
    warmGlow: "rgba(255, 214, 0, 0.10)",
    noiseOpacity: 0,
  },
  typography: {
    fontSans: `"Inter", ${SYSTEM_SANS}`,
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontDisplay: `"Space Grotesk", "Inter", ${SYSTEM_SANS}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap",
    baseSize: "15px",
    lineHeight: "1.65",
    letterSpacing: "0",
  },
  layout: {
    radius: "0",
    density: "comfortable",
  },
  layoutVariant: "tiled",
  colorOverrides: {
    card: "#F5F0E8",
    cardForeground: "#1A1A1A",
    popover: "#FAF8F5",
    popoverForeground: "#1A1A1A",
    primary: "#FFD600",
    primaryForeground: "#1A1A1A",
    secondary: "#FFFFFF",
    secondaryForeground: "#1A1A1A",
    muted: "#F0EDE8",
    mutedForeground: "#3A3A3A",
    accent: "#FFD600",
    accentForeground: "#1A1A1A",
    destructive: "#FF6B6B",
    destructiveForeground: "#1A1A1A",
    success: "#00A070",
    warning: "#B45309",
    border: "#1A1A1A",
    input: "#1A1A1A",
    ring: "#2979FF",
  },
  componentStyles: {
    card: {
      border: "3px solid #1A1A1A",
      background: "#F5F0E8",
      boxShadow: "5px 5px 0 0 #1A1A1A",
    },
    header: {
      background: "#FAF8F5",
    },
    sidebar: {
      background: "#FAF8F5",
    },
    backdrop: {
      // Memphis canvas is light: kill all the dark-theme blend modes that
      // muddy a cream background. Difference + color-dodge + lighten only
      // make sense over a deep teal/black canvas.
      canvasBlendMode: "normal",
      glowBlendMode: "normal",
      glowOpacity: "0",
      noiseBlendMode: "multiply",
      noiseOpacity: "0.04",
      fillerOpacity: "0",
      fillerBlendMode: "normal",
      backgroundSize: "40px 40px",
      backgroundPosition: "0 0",
    },
  },
  assets: {
    bg: "linear-gradient(rgba(26,26,26,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(26,26,26,0.06) 1px, transparent 1px)",
  },
  customCSS: `
:root {
  color-scheme: light;
  --hy-ink: #1A1A1A;
  --hy-paper: #FAF8F5;
  --hy-panel: #F5F0E8;
  --hy-panel-hi: #FFFFFF;
  --hy-yellow: #FFD600;
  --hy-blue: #2979FF;
  --hy-coral: #FF6B6B;
  --hy-mint: #00E5A0;
  --hy-purple: #7C4DFF;
}

body {
  background: var(--hy-paper);
  color: var(--hy-ink);
}

/* ──────────────────────────────────────────────────────────────
   Root layout overrides — App.tsx hardcodes
     bg-black uppercase text-midground font-mondwest
   on the root div. For the Memphis×Brutalist lens we flip the
   canvas to cream, drop the uppercase mask, and swap the display
   font back to Space Grotesk so headings and body text stay
   legible. Scoped to [data-layout-variant="tiled"] so other
   themes keep their look.
   ────────────────────────────────────────────────────────────── */
#root [data-layout-variant="tiled"] {
  background: var(--hy-paper) !important;
  color: var(--hy-ink) !important;
  text-transform: none !important;
  font-family: "Inter", system-ui, sans-serif !important;
}

/* Body text and generic text nodes inside the tiled shell keep
   their natural case — no global uppercase. */
#root [data-layout-variant="tiled"] *:not(.font-mono):not(code):not(pre) {
  text-transform: none;
}

/* Subtle Memphis grid backdrop on the main canvas. */
#root [data-layout-variant="tiled"] main {
  background-image:
    linear-gradient(rgba(26,26,26,0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(26,26,26,0.055) 1px, transparent 1px);
  background-size: 40px 40px;
}

#root [data-layout-variant="tiled"] header,
#root [data-layout-variant="tiled"] aside {
  background: var(--hy-paper) !important;
  color: var(--hy-ink) !important;
  border-color: var(--hy-ink) !important;
}

#root [data-layout-variant="tiled"] aside {
  border-right: 3px solid var(--hy-ink) !important;
}

#root [data-layout-variant="tiled"] header {
  border-bottom: 3px solid var(--hy-ink) !important;
}

/* Cards/panels: cream panel with hard ink border and offset shadow. */
#root [data-layout-variant="tiled"] [class*="bg-card"],
#root [data-layout-variant="tiled"] [class*="bg-popover"] {
  background: var(--hy-panel) !important;
  color: var(--hy-ink) !important;
  border: 3px solid var(--hy-ink) !important;
  box-shadow: 5px 5px 0 0 var(--hy-ink);
}

#root [data-layout-variant="tiled"] [class*="bg-muted"],
#root [data-layout-variant="tiled"] [class*="bg-secondary"] {
  background: var(--hy-panel) !important;
  color: var(--hy-ink) !important;
}

#root [data-layout-variant="tiled"] main,
#root [data-layout-variant="tiled"] [class*="bg-card"],
#root [data-layout-variant="tiled"] [class*="bg-muted"],
#root [data-layout-variant="tiled"] [class*="bg-secondary"],
#root [data-layout-variant="tiled"] [class*="bg-popover"] {
  border-radius: 0 !important;
}

#root [data-layout-variant="tiled"] button,
#root [data-layout-variant="tiled"] [role="button"],
#root [data-layout-variant="tiled"] a,
#root [data-layout-variant="tiled"] input,
#root [data-layout-variant="tiled"] textarea,
#root [data-layout-variant="tiled"] select {
  border-radius: 0 !important;
}

/* Primary buttons: ink text on yellow with the hard-shadow signature. */
#root [data-layout-variant="tiled"] button:not([aria-label]):not(:disabled),
#root [data-layout-variant="tiled"] [role="button"]:not(:disabled) {
  font-family: "JetBrains Mono", monospace;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 2px solid var(--hy-ink);
  background: var(--hy-panel-hi);
  color: var(--hy-ink);
  box-shadow: 3px 3px 0 0 var(--hy-ink);
  transition: transform 0.08s ease, box-shadow 0.08s ease;
}

#root [data-layout-variant="tiled"] button[class*="bg-primary"]:not(:disabled),
#root [data-layout-variant="tiled"] [role="button"][class*="bg-primary"]:not(:disabled) {
  background: var(--hy-yellow) !important;
  color: var(--hy-ink) !important;
}

#root [data-layout-variant="tiled"] button:not(:disabled):hover,
#root [data-layout-variant="tiled"] [role="button"]:not(:disabled):hover {
  box-shadow: 5px 5px 0 0 var(--hy-ink);
  transform: translate(-1px, -1px);
}

#root [data-layout-variant="tiled"] button:not(:disabled):active,
#root [data-layout-variant="tiled"] [role="button"]:not(:disabled):active {
  box-shadow: 1px 1px 0 0 var(--hy-ink);
  transform: translate(2px, 2px);
}

/* Inputs: cream field with ink border. */
#root [data-layout-variant="tiled"] input,
#root [data-layout-variant="tiled"] textarea,
#root [data-layout-variant="tiled"] select {
  background: var(--hy-panel-hi) !important;
  color: var(--hy-ink) !important;
  border: 2px solid var(--hy-ink) !important;
}

#root [data-layout-variant="tiled"] input:focus-visible,
#root [data-layout-variant="tiled"] textarea:focus-visible,
#root [data-layout-variant="tiled"] select:focus-visible {
  outline: 3px solid var(--hy-blue);
  outline-offset: 2px;
  box-shadow: 3px 3px 0 0 var(--hy-ink);
}

#root [data-layout-variant="tiled"] .border,
#root [data-layout-variant="tiled"] [class*="border-border"],
#root [data-layout-variant="tiled"] [class*="border-current"] {
  border-color: var(--hy-ink) !important;
}

/* Headings use Space Grotesk Display for Memphis punch, with
   extra weight so they read against the grid background. */
#root [data-layout-variant="tiled"] h1,
#root [data-layout-variant="tiled"] h2,
#root [data-layout-variant="tiled"] h3,
#root [data-layout-variant="tiled"] h4 {
  font-family: "Space Grotesk", Inter, sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.015em !important;
  color: var(--hy-ink) !important;
  text-transform: none !important;
}

#root [data-layout-variant="tiled"] h1 { font-weight: 800 !important; }

/* Muted text: keep ink but at 60% — readable on cream while
   preserving hierarchy. */
#root [data-layout-variant="tiled"] [class*="text-muted-foreground"] {
  color: color-mix(in srgb, var(--hy-ink) 60%, transparent) !important;
}

/* Active nav pill: ink fill with cream text — Memphis pill style. */
#root [data-layout-variant="tiled"] a[aria-current="page"],
#root [data-layout-variant="tiled"] a.active {
  background: var(--hy-ink) !important;
  color: var(--hy-paper) !important;
  border: 2px solid var(--hy-ink) !important;
}

#root [data-layout-variant="tiled"] code,
#root [data-layout-variant="tiled"] pre,
#root [data-layout-variant="tiled"] .font-mono,
#root [data-layout-variant="tiled"] .font-mono-ui {
  font-family: "JetBrains Mono", monospace;
  color: var(--hy-ink);
}

#root [data-layout-variant="tiled"] pre {
  background: var(--hy-panel-hi) !important;
  border: 2px solid var(--hy-ink) !important;
  box-shadow: 3px 3px 0 0 var(--hy-ink);
  padding: 0.75rem 1rem;
}

/* Kill the default dark-theme backdrop warm-glow layer so the
   cream canvas stays clean. The Backdrop <div> renders radial
   gradients that overlay the page even when the theme's glow
   colour is translucent. */
#root [data-layout-variant="tiled"] .theme-default-filler,
#root [data-layout-variant="tiled"] [class*="backdrop"] > img {
  display: none !important;
}

/* Tame badge grain overlay — currentColor over cream produces
   a smudgy haze.  Memphis badges should be hard-edged.        */
#root [data-layout-variant="tiled"] .grain::after {
  display: none;
}

#root [data-layout-variant="tiled"] ::selection {
  background: var(--hy-yellow);
  color: var(--hy-ink);
}

#root [data-layout-variant="tiled"] ::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

#root [data-layout-variant="tiled"] ::-webkit-scrollbar-track {
  background: var(--hy-paper);
}

#root [data-layout-variant="tiled"] ::-webkit-scrollbar-thumb {
  background: var(--hy-ink);
  border-radius: 0;
}
`,
};

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  haoyuanlin: haoyuanlinTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
};
