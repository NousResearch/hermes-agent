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

export const paperlightTheme: DashboardTheme = {
  name: "paperlight",
  label: "Paperlight",
  description: "Light paper workspace with high-contrast text",
  palette: {
    background: { hex: "#f7f1e5", alpha: 1 },
    midground: { hex: "#0b1220", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0.28 },
    warmGlow: "rgba(245, 174, 84, 0.18)",
    noiseOpacity: 0.16,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Source Sans 3", "Avenir Next", "Segoe UI", ${SYSTEM_SANS}`,
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontDisplay: `"Source Serif 4", Georgia, serif`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
    lineHeight: "1.58",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "0.85rem",
  },
  assets: {
    bg: "radial-gradient(circle at 12% 8%, rgba(255, 208, 128, 0.26), transparent 32%), radial-gradient(circle at 90% 4%, rgba(111, 143, 205, 0.18), transparent 30%), linear-gradient(135deg, #fbf7ef 0%, #f2eadb 48%, #e9dfcd 100%)",
  },
  componentStyles: {
    page: {
      background:
        "linear-gradient(180deg, rgba(255, 252, 245, 0.96), rgba(244, 236, 222, 0.92))",
    },
    card: {
      background:
        "linear-gradient(180deg, rgba(255, 253, 248, 0.94), rgba(249, 244, 235, 0.90))",
      boxShadow:
        "0 18px 48px -34px rgba(29, 42, 62, 0.42), inset 0 0 0 1px rgba(96, 77, 46, 0.11)",
    },
    header: {
      background:
        "linear-gradient(180deg, rgba(255, 253, 248, 0.96), rgba(246, 239, 226, 0.92))",
    },
    sidebar: {
      background:
        "linear-gradient(180deg, rgba(255, 253, 248, 0.94), rgba(241, 232, 216, 0.92))",
    },
    tab: {
      background: "rgba(255, 255, 255, 0.58)",
    },
    badge: {
      background: "rgba(42, 83, 130, 0.10)",
    },
    backdrop: {
      backgroundSize: "cover",
      backgroundPosition: "center",
      fillerOpacity: "0.72",
      fillerBlendMode: "normal",
    },
  },
  colorOverrides: {
    card: "#fffaf1",
    cardForeground: "#0b1220",
    popover: "#fffdf8",
    popoverForeground: "#0b1220",
    primary: "#245a92",
    primaryForeground: "#ffffff",
    secondary: "#eadcc5",
    secondaryForeground: "#111827",
    muted: "#eee3d1",
    mutedForeground: "#303846",
    accent: "#d88325",
    accentForeground: "#20140a",
    destructive: "#b42318",
    destructiveForeground: "#ffffff",
    success: "#20744f",
    warning: "#c57a14",
    border: "#d6c7af",
    input: "#d1c1a8",
    ring: "#245a92",
  },
  customCSS: `
    :root body {
      color-scheme: light;
      color: #0b1220;
    }

    :root body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      opacity: 0.22;
      background-image:
        linear-gradient(rgba(80, 65, 42, 0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(80, 65, 42, 0.04) 1px, transparent 1px);
      background-size: 32px 32px, 32px 32px;
    }

    :root .text-muted-foreground {
      color: #303846;
    }

    :root .border-border {
      border-color: rgba(114, 94, 62, 0.22);
    }

    :root input,
    :root textarea,
    :root select {
      background-color: rgba(255, 253, 248, 0.9);
    }

    :root code,
    :root pre {
      background-color: rgba(36, 90, 146, 0.08);
      color: #07111f;
    }
  `,
};

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  default: defaultTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
  paperlight: paperlightTheme,
};
