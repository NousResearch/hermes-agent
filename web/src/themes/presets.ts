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

export const cleanTheme: DashboardTheme = {
  name: "clean",
  label: "Clean",
  description: "Clean, calm, readable light theme for the Hermes dashboard.",
  palette: {
    background: { hex: "#f4f1e9", alpha: 1 },
    midground: { hex: "#1f3f37", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0.06 },
    warmGlow: "rgba(47, 143, 122, 0.06)",
    noiseOpacity: 0,
  },
  typography: {
    fontSans: `"IBM Plex Sans", Inter, ${SYSTEM_SANS}`,
    fontMono: `"IBM Plex Mono", "JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap",
    baseSize: "14px",
    lineHeight: "1.55",
    letterSpacing: "0",
  },
  layout: {
    radius: "0.7rem",
    density: "comfortable",
  },
  layoutVariant: "standard",
  colorOverrides: {
    card: "#fffefa",
    cardForeground: "#1d2926",
    popover: "#fffefa",
    popoverForeground: "#1d2926",
    primary: "#287d6b",
    primaryForeground: "#ffffff",
    secondary: "#e7eee9",
    secondaryForeground: "#203b34",
    muted: "#eef1ec",
    mutedForeground: "#5e6c67",
    accent: "#dcebe4",
    accentForeground: "#1f3f37",
    destructive: "#b4403e",
    destructiveForeground: "#ffffff",
    success: "#287d6b",
    warning: "#8a650f",
    border: "#d3d9d4",
    input: "#bccbc4",
    ring: "#5c9f8d",
  },
  componentStyles: {
    page: {
      background: "#f4f1e9",
    },
    card: {
      background: "#fffefa",
      border: "1px solid #cbd5cf",
      boxShadow: "0 8px 24px rgba(29, 41, 38, 0.06)",
    },
    header: {
      background: "rgba(244, 241, 233, 0.96)",
      borderBottom: "1px solid #d7ddd8",
      backdropFilter: "blur(12px) saturate(120%)",
    },
    sidebar: {
      background: "#efede6",
      borderRight: "1px solid #d7ddd8",
    },
    badge: {
      borderRadius: "999px",
    },
    tab: {
      borderRadius: "0.55rem",
    },
    backdrop: {
      opacity: "0",
      fillerOpacity: "0",
      fillerBlendMode: "normal",
    },
  },
  customCSS: `
    :root {
      color-scheme: light;
      --clean-bg: #f4f1e9;
      --clean-panel: #fffefa;
      --clean-sidebar: #efede6;
      --clean-text: #1d2926;
      --clean-heading: #153a33;
      --clean-muted: #52665f;
      --clean-border: #cbd5cf;
      --clean-accent: #287d6b;
      --clean-accent-soft: #dcebe4;
      --clean-danger: #b4403e;
    }

    html,
    body,
    #root,
    [data-layout-variant] {
      background: var(--clean-bg) !important;
      color: var(--clean-text) !important;
      font-family: var(--theme-font-sans) !important;
      text-transform: none !important;
      letter-spacing: normal !important;
      text-shadow: none !important;
      accent-color: var(--clean-accent) !important;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    [class*="bg-black"],
    .bg-black,
    main,
    main > div,
    [class*="bg-background"],
    [class*="bg-background-base"] {
      background-color: var(--clean-bg) !important;
    }

    [style*="mix-blend-mode"],
    [style*="mixBlendMode"],
    .theme-default-filler,
    img.theme-default-filler,
    .pointer-events-none.fixed.inset-0[style*="mix-blend-mode"] {
      mix-blend-mode: normal !important;
      opacity: 0 !important;
      visibility: hidden !important;
    }

    *,
    [class*="uppercase"],
    [class*="tracking-"] {
      letter-spacing: normal !important;
      text-transform: none !important;
      text-shadow: none !important;
    }

    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    [class*="font-expanded"] {
      color: var(--clean-heading) !important;
      font-family: var(--theme-font-sans) !important;
      font-weight: 700 !important;
    }

    h1 { font-size: 1.45rem !important; }
    h2 { font-size: 1.18rem !important; }
    h3 { font-size: 1.02rem !important; }

    aside,
    [data-sidebar],
    nav[class*="border-r"],
    .border-r {
      background-color: var(--clean-sidebar) !important;
      border-color: #d7ddd8 !important;
    }

    aside nav a > span[aria-hidden] {
      display: none !important;
      opacity: 0 !important;
      background: transparent !important;
    }

    aside nav ul li a:not(.text-midground),
    aside nav ul li a:not(.text-midground):hover,
    aside nav ul li a:not(.text-midground):focus,
    aside nav ul li a:not(.text-midground):visited {
      background: transparent !important;
      color: #123b34 !important;
      box-shadow: none !important;
      opacity: 1 !important;
    }

    aside nav ul li a.text-midground {
      background: var(--clean-accent-soft) !important;
      color: #0f332d !important;
      box-shadow: inset 3px 0 0 var(--clean-accent) !important;
      opacity: 1 !important;
    }

    aside nav a svg,
    aside nav a svg *,
    aside nav a > span:not([aria-hidden]) {
      background: transparent !important;
      color: currentColor !important;
      opacity: 1 !important;
    }

    header,
    header h1,
    header h2,
    header span,
    header div,
    .text-midground,
    [class*="text-midground"] {
      color: var(--clean-heading) !important;
    }

    .text-muted-foreground,
    [class*="text-muted-foreground"],
    [class*="text-muted"],
    [class*="text-midground/"],
    [class*="text-current/"] {
      color: var(--clean-muted) !important;
      opacity: 1 !important;
    }

    [class*="bg-card"],
    [class*="bg-popover"],
    [class*="bg-background/40"],
    [class*="bg-background/50"],
    [class*="bg-background-base/40"],
    [class*="bg-background-base/50"] {
      background-color: var(--clean-panel) !important;
      color: var(--clean-text) !important;
    }

    .border,
    .border-border,
    [class*="border-border"],
    [class*="rounded"] {
      border-color: var(--clean-border) !important;
    }

    main [class*="border-b"],
    main [class*="border-b"] * {
      background: #fbfaf6 !important;
      border-color: var(--clean-border) !important;
      color: #24453e !important;
      mix-blend-mode: normal !important;
      opacity: 1 !important;
    }

    input,
    select,
    textarea,
    [role="combobox"],
    input[class*="text-"],
    textarea[class*="text-"],
    select[class*="text-"] {
      min-height: 2.45rem !important;
      background: #fffefa !important;
      border: 1px solid #bccbc4 !important;
      border-radius: 0.65rem !important;
      color: #172b27 !important;
      font-family: var(--theme-font-sans) !important;
      font-size: 0.95rem !important;
      line-height: 1.45 !important;
    }

    input::placeholder,
    textarea::placeholder {
      color: #62756f !important;
      opacity: 1 !important;
    }

    button,
    [role="button"] {
      border-radius: 0.65rem !important;
      font-family: var(--theme-font-sans) !important;
      font-weight: 600 !important;
      letter-spacing: normal !important;
      text-transform: none !important;
    }

    button:not([class*="bg-primary"]):not([class*="text-primary"]),
    [role="button"]:not([class*="bg-primary"]):not([class*="text-primary"]) {
      background-color: #f8f7f2 !important;
      border-color: #c9d2cc !important;
      color: #183c35 !important;
    }

    button[class*="bg-primary"],
    [class*="bg-primary"] {
      background: var(--clean-accent) !important;
      border-color: #216b5c !important;
      color: #ffffff !important;
    }

    code,
    pre,
    kbd,
    samp,
    .font-mono,
    [class*="font-mono"] {
      font-family: var(--theme-font-mono) !important;
      letter-spacing: 0 !important;
    }

    small,
    [class*="text-[0.65rem]"],
    [class*="text-[0.7rem]"],
    [class*="text-[0.75rem]"],
    [class*="text-xs"]:not(input):not(textarea):not(select) {
      color: var(--clean-muted) !important;
      font-size: 0.8rem !important;
      line-height: 1.35 !important;
      opacity: 1 !important;
    }

    mark,
    mark.bg-warning\\/30,
    mark.text-warning,
    .bg-warning\\/30,
    .text-warning.bg-warning\\/30 {
      background: #d7eec7 !important;
      color: #25493f !important;
      border-radius: 0.18rem !important;
      box-decoration-break: clone !important;
      -webkit-box-decoration-break: clone !important;
      mix-blend-mode: normal !important;
      text-shadow: none !important;
    }

    ::selection,
    *::selection,
    input::selection,
    textarea::selection,
    code::selection,
    pre::selection {
      background: #b8d8cc !important;
      color: #08241f !important;
      text-shadow: none !important;
    }

    ::-moz-selection,
    *::-moz-selection,
    input::-moz-selection,
    textarea::-moz-selection,
    code::-moz-selection,
    pre::-moz-selection {
      background: #b8d8cc !important;
      color: #08241f !important;
      text-shadow: none !important;
    }
  `,
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

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  clean: cleanTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
};
