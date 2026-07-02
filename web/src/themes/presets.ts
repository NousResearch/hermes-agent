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

/** Light mode — vivid Nous-blue accents on a cream canvas. */
export const nousBlueTheme: DashboardTheme = {
  name: "nous-blue",
  label: "Nous Blue",
  description: "Light mode — vivid Nous-blue accents on cream canvas",
  palette: {
    background: { hex: "#E8F2FD", alpha: 1 },
    midground: { hex: "#0053FD", alpha: 1 },
    foreground: { hex: "#170d02", alpha: 0 },
    warmGlow: "rgba(0, 83, 253, 0.12)",
    noiseOpacity: 0,
  },
  typography: DEFAULT_TYPOGRAPHY,
  layout: DEFAULT_LAYOUT,
  terminalBackground: "#f5f8fc",
  terminalForeground: "#170d02",
  seriesColors: {
    inputTokenAccent: "#001934",
    outputTokenAccent: "#0053fd",
  },
  swatchColors: ["#170d02", "#0053FD", "#E8F2FD"],
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

/**
 * Aurora — dark frosted-glass theme. Translucent surfaces with backdrop-blur
 * over a gold + indigo aurora backdrop, unified Inter typography (mono kept
 * only for technical tokens), silver display text, and high-contrast, readable
 * tables and menus. Contributed by Michel Marrazzo (KuramaLab).
 */
export const auroraTheme: DashboardTheme = {
  name: "aurora",
  label: "Aurora",
  description:
    "Dark frosted-glass — translucent surfaces over a gold & indigo aurora, unified Inter type, high-contrast tables and menus",
  palette: {
    background: { hex: "#141518", alpha: 1 },
    midground: { hex: "#ecedef", alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: "rgba(232, 160, 0, 0.12)",
    noiseOpacity: 0,
  },
  typography: {
    ...DEFAULT_TYPOGRAPHY,
    fontSans: `"Inter", system-ui, -apple-system, "Segoe UI", sans-serif`,
    fontMono: `"JetBrains Mono", ui-monospace, "SF Mono", monospace`,
    fontUrl:
      "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap",
  },
  layout: {
    ...DEFAULT_LAYOUT,
    radius: "1rem",
  },
  layoutVariant: "standard",
  colorOverrides: {
    card: "#31333a",
    cardForeground: "#ecedef",
    popover: "#3a3c43",
    popoverForeground: "#f4f5f7",
    primary: "#e8a000",
    primaryForeground: "#141414",
    secondary: "#2a2b31",
    secondaryForeground: "#f0d69a",
    muted: "#2a2b31",
    mutedForeground: "#a6a8ae",
    accent: "#e8a000",
    accentForeground: "#141414",
    destructive: "#f0616d",
    destructiveForeground: "#141414",
    success: "#34d399",
    warning: "#f5b301",
    border: "#33343c",
    input: "#33343c",
    ring: "#e8a000",
  },
  componentStyles: {
    card: {
      background: "rgba(40, 43, 52, 0.55)",
      border: "1px solid rgba(255,255,255,0.09)",
      boxShadow:
        "inset 0 1px 0 rgba(255,255,255,.07), 0 20px 50px rgba(0,0,0,.45)",
    },
    header: {
      background: "rgba(20, 21, 26, 0.55)",
      boxShadow: "inset 0 -1px 0 rgba(255,255,255,.07)",
    },
    sidebar: {
      background: "rgba(21, 22, 27, 0.72)",
      boxShadow: "inset -1px 0 0 rgba(255,255,255,.06)",
    },
    badge: {
      background: "rgba(255,255,255,.06)",
    },
    page: {
      background: "transparent",
    },
  },
  swatchColors: ["#141518", "#e8a000", "#31333a"],
  terminalBackground: "#0f1014",
  terminalForeground: "#ecedef",
  customCSS: String.raw`
  /* === Unified typography: one voice (Inter) + mono only for technical tokens === */
  /* Hermes' UI mixes 5 families (Mondwest, Rules Compressed/Expanded, system-sans, mono).
     Remap the 3 display fonts onto the theme sans so the whole UI becomes Inter.
     font-mono / font-mono-ui stay JetBrains Mono (via typography.fontMono) for model ids, tools, version. */
  :root {
    --font-mondwest:         var(--theme-font-sans);
    --font-rules-compressed: var(--theme-font-sans);
    --font-rules-expanded:   var(--theme-font-sans);
  }
  /* Tracking: display fonts used wide 0.08-0.2em spacing that reads oddly in Inter -> tighten */
  .tracking-\[0\.2em\],
  .tracking-\[0\.12em\],
  .tracking-\[0\.1em\],
  .tracking-\[0\.08em\] { letter-spacing: .02em !important; }
  .text-display { letter-spacing: .02em; }
  /* Size hierarchy: card titles jumped to text-base (bigger than the page title) -> realign */
  .flex-col.gap-1\.5.p-4.border-b .text-base { font-size: .92rem !important; }
  /* Tiny footer text (0.65rem) -> more legible */
  .text-\[0\.65rem\] { font-size: .72rem !important; }

  /* === Glass / frosted === */
  /* Aurora backdrop: gold (brand) + indigo glows on a near-black base, giving
     depth for the glass backdrop-blur to diffuse. */
  body {
    background:
      radial-gradient(1100px 620px at 12% -8%, rgba(232,160,0,.12), transparent 60%),
      radial-gradient(1000px 700px at 100% 0%, rgba(96,124,255,.10), transparent 55%),
      radial-gradient(900px 900px at 60% 120%, rgba(232,160,0,.06), transparent 60%),
      #0f1014 !important;
    background-attachment: fixed !important;
  }

  /* Card hover: light up the glass */
  [class*="card"], .card { transition: box-shadow .2s ease, border-color .2s ease, background .2s ease; }
  [class*="card"]:hover {
    border-color: rgba(232,160,0,.35) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 26px 64px rgba(0,0,0,.5) !important;
  }

  /* Readable tables: distinct header + gold rule, separated rows, zebra, gold hover */
  table { border-collapse: separate; border-spacing: 0; width: 100%; border-radius: .6rem; overflow: hidden; }
  thead th, [role="columnheader"] {
    background: #26272e !important;
    color: #f4f5f7 !important;
    font-weight: 600;
    border-bottom: 2px solid rgba(232,160,0,.7) !important;
    text-align: left;
  }
  tbody td, [role="cell"], [role="gridcell"] {
    border-bottom: 1px solid #2a2b31 !important;
    color: #dfe1e6 !important;
  }
  tbody tr:nth-child(even) td { background: rgba(255,255,255,.022); }
  tbody tr:hover td { background: rgba(232,160,0,.12) !important; }
  tbody tr:last-child td { border-bottom: none !important; }

  /* Same-looking lists -> visible divider */
  [class*="divide-y"] > * + * { border-top: 1px solid #2a2b31 !important; }

  /* Dialogs / popovers / menus: elevated glass */
  [role="dialog"], [data-radix-popper-content-wrapper] > * {
    background: rgba(34,37,46,.72) !important;
    -webkit-backdrop-filter: blur(22px) saturate(1.4) !important;
    backdrop-filter: blur(22px) saturate(1.4) !important;
    border-radius: 1rem !important;
    border: 1px solid rgba(255,255,255,.10) !important;
    box-shadow: 0 24px 64px rgba(0,0,0,.62) !important;
  }

  /* Inputs: dark glass with visible border + gold focus */
  input, textarea, select {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    color: #ecedef !important;
    border-radius: .6rem !important;
  }
  input:focus, textarea:focus, select:focus { border-color: #e8a000 !important; box-shadow: 0 0 0 3px rgba(232,160,0,.28) !important; }

  /* Links / text accents in gold */
  a { color: #f0b62e; }
  a:hover { color: #ffc540; }

  /* Lift dim tones for legibility */
  .text-muted-foreground, [class*="muted-foreground"] { color: #a6a8ae !important; }
  .opacity-50, .opacity-40 { opacity: .72 !important; }

  /* Badges (tui/cli/telegram, counts): rounded and readable (font/tracking unified above) */
  .font-compressed {
    border-radius: .5rem !important;
    padding: .18rem .55rem !important;
    font-weight: 600 !important;
    text-transform: none !important;
  }
  /* outline variant: glass pill with luminous border */
  .font-compressed.bg-transparent {
    color: #e6e8ed !important;
    background: rgba(255,255,255,.05) !important;
    border-color: rgba(255,255,255,.14) !important;
    -webkit-backdrop-filter: blur(6px) !important;
    backdrop-filter: blur(6px) !important;
  }

  /* Card = frosted glass, rounded, clips its content (header included) */
  .border.bg-background-base\/80 {
    border-radius: 1rem !important;
    overflow: hidden !important;
    background: rgba(40, 43, 52, 0.55) !important;
    -webkit-backdrop-filter: blur(18px) saturate(1.3) !important;
    backdrop-filter: blur(18px) saturate(1.3) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.07), 0 20px 50px rgba(0,0,0,.45) !important;
  }

  /* Panel header (CardHeader): darker glass bar; targets the common denominator of every CardHeader */
  .flex-col.gap-1\.5.p-4.border-b {
    background: linear-gradient(180deg, rgba(22,24,30,.55), rgba(12,13,17,.28)) !important;
    border-bottom: 1px solid rgba(255,255,255,.07) !important;
  }
  /* header text -> silver (title = font-expanded, subtitle = font-mondwest) */
  .flex-col.gap-1\.5.p-4.border-b .font-expanded {
    color: #d6d9de !important;
    -webkit-text-fill-color: #d6d9de !important;
    font-weight: 600;
  }
  .flex-col.gap-1\.5.p-4.border-b .font-mondwest {
    color: #b9bdc4 !important;
    -webkit-text-fill-color: #b9bdc4 !important;
  }

  /* Sidebar menu: compact size (tracking already tightened above) */
  .text-display.uppercase.tracking-\[0\.12em\] { font-size: .8rem !important; }

  /* Scrollbars: dark but visible */
  * { scrollbar-color: #4a4b54 transparent; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-thumb { background: #4a4b54; border-radius: 8px; border: 2px solid transparent; background-clip: content-box; }
  ::-webkit-scrollbar-thumb:hover { background: #e8a000; background-clip: content-box; }
`,
};

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  "nous-blue": nousBlueTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
  aurora: auroraTheme,
};
