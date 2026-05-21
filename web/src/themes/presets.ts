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

export const makimaTheme: DashboardTheme = {
  name: "makima",
  label: "Makima",
  description: "Warm Makima desktop palette for the Hermes dashboard.",
  palette: {
    background: { hex: "#1d2021", alpha: 1 },
    midground: { hex: "#282828", alpha: 0.96 },
    foreground: { hex: "#f0e8d0", alpha: 0.08 },
    warmGlow: "rgba(215, 153, 33, 0.34)",
    noiseOpacity: 0.75,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.65rem",
    density: "comfortable",
  },
  colorOverrides: {
    card: "#282828",
    cardForeground: "#f0e8d0",
    popover: "#1d2021",
    popoverForeground: "#fbf1c7",
    primary: "#d79921",
    primaryForeground: "#282828",
    secondary: "#504945",
    secondaryForeground: "#f0e8d0",
    muted: "#504945",
    mutedForeground: "#a89984",
    accent: "#d3869b",
    accentForeground: "#282828",
    destructive: "#cc241d",
    destructiveForeground: "#fbf1c7",
    success: "#b8bb26",
    warning: "#fabd2f",
    border: "#5d141a",
    input: "#504945",
    ring: "#d79921",
  },
  componentStyles: {
    card: {
      background: "linear-gradient(135deg, rgba(40,40,40,0.96), rgba(29,32,33,0.98))",
      borderColor: "rgba(215,153,33,0.45)",
      boxShadow: "0 0 0 1px rgba(93,20,26,0.35), 0 20px 60px rgba(29,32,33,0.55)",
    },
    header: {
      background: "linear-gradient(90deg, rgba(93,20,26,0.82), rgba(40,40,40,0.78))",
      borderColor: "rgba(215,153,33,0.45)",
    },
    sidebar: {
      background: "rgba(29,32,33,0.92)",
      borderColor: "rgba(80,73,69,0.85)",
    },
    badge: {
      background: "rgba(215,153,33,0.16)",
      borderColor: "rgba(215,153,33,0.5)",
    },
    progress: {
      background: "linear-gradient(90deg, #5d141a, #d79921, #fabd2f)",
    },
  },
};

export const manjaroTheme: DashboardTheme = {
  name: "manjaro",
  label: "Manjaro",
  description: "Midnight purple and magenta Manjaro desktop palette for the Hermes dashboard.",
  palette: {
    background: { hex: "#090714", alpha: 1 },
    midground: { hex: "#0d0b1a", alpha: 0.96 },
    foreground: { hex: "#e8e0f0", alpha: 0.08 },
    warmGlow: "rgba(230, 32, 122, 0.38)",
    noiseOpacity: 0.82,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.75rem",
    density: "comfortable",
  },
  colorOverrides: {
    card: "#0d0b1a",
    cardForeground: "#e8e0f0",
    popover: "#090714",
    popoverForeground: "#f5f0ff",
    primary: "#e6207a",
    primaryForeground: "#f5f0ff",
    secondary: "#1a1530",
    secondaryForeground: "#e8e0f0",
    muted: "#1a1530",
    mutedForeground: "#6e6580",
    accent: "#7b2fbe",
    accentForeground: "#f5f0ff",
    destructive: "#ff1493",
    destructiveForeground: "#f5f0ff",
    success: "#2eb398",
    warning: "#f0c040",
    border: "#7b2fbe",
    input: "#1a1530",
    ring: "#e6207a",
  },
  componentStyles: {
    card: {
      background: "linear-gradient(135deg, rgba(13,11,26,0.96), rgba(26,21,48,0.88))",
      borderColor: "rgba(230,32,122,0.48)",
      boxShadow: "0 0 0 1px rgba(123,47,190,0.3), 0 24px 70px rgba(9,7,20,0.65)",
    },
    header: {
      background: "linear-gradient(90deg, rgba(230,32,122,0.78), rgba(123,47,190,0.5), rgba(13,11,26,0.8))",
      borderColor: "rgba(92,200,216,0.38)",
    },
    sidebar: {
      background: "rgba(9,7,20,0.92)",
      borderColor: "rgba(123,47,190,0.65)",
    },
    badge: {
      background: "rgba(230,32,122,0.16)",
      borderColor: "rgba(230,32,122,0.52)",
    },
    progress: {
      background: "linear-gradient(90deg, #1a237e, #7b2fbe, #e6207a, #ff1493)",
    },
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
  makima: makimaTheme,
  manjaro: manjaroTheme,
};
