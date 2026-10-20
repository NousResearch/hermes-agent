import { parseColor, THEME_PRESET_PALETTES, type ThemePresetPalette } from "@hermes/shared";
import type { DashboardTheme, ThemePalette, ThemeTypography, ThemeLayout } from "./types";

/**
 * Built-in dashboard themes.
 *
 * Each theme defines its own palette, typography, and layout so switching
 * themes produces visible changes beyond just color — fonts, density, and
 * corner-radius all shift to match the theme's personality.
 *
 * Theme names must stay in sync with the backend's
 * `_BUILTIN_DASHBOARD_THEMES` list in `hermes_cli/web_server.py`.
 *
 * Presets that also ship on the desktop (midnight, ember, mono, cyberpunk)
 * take their colours from `@hermes/shared` `THEME_PRESET_PALETTES` so both
 * surfaces render one palette; only typography/layout/overrides live here.
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

/**
 * Project a shared (desktop-shaped) preset palette onto the dashboard's
 * 3-slot model. The dashboard's `midground` is its text + primary-fill
 * colour, which is the desktop's `primary`; its `warmGlow` is the brand
 * accent stroke, which is the desktop's `midground` (falling back to `ring`).
 * `foreground` stays the dashboard's invisible white overlay. Dark palettes
 * are the dashboard's home turf, so a preset shipping `darkColors` is read
 * from that side.
 */
export function webPresetFromShared(
  preset: ThemePresetPalette,
): Omit<ThemePalette, "noiseOpacity"> {
  const colors = preset.darkColors ?? preset.colors;
  const [r, g, b] = parseColor(colors.midground ?? colors.ring) ?? [255, 255, 255];
  return {
    background: { hex: colors.background, alpha: 1 },
    midground: { hex: colors.primary, alpha: 1 },
    foreground: { hex: "#ffffff", alpha: 0 },
    warmGlow: `rgba(${r}, ${g}, ${b}, 0.3)`,
  };
}

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
    ...webPresetFromShared(THEME_PRESET_PALETTES.midnight),
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
    ...webPresetFromShared(THEME_PRESET_PALETTES.ember),
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
    ...webPresetFromShared(THEME_PRESET_PALETTES.mono),
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
    ...webPresetFromShared(THEME_PRESET_PALETTES.cyberpunk),
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

// ---------------------------------------------------------------------------
// Palette themes — full-surface palettes (canvas, cards, accent, borders)
// ---------------------------------------------------------------------------

/** Surface tokens for a palette theme; each maps onto the dashboard's slots. */
interface PaletteTokens {
  /** Page canvas. */
  bg: string;
  /** Cards, popovers. */
  panel: string;
  /** Muted fills: secondary buttons, hover rows, tabs. */
  panelSoft: string;
  /** Body text. */
  ink: string;
  /** Secondary text. */
  muted: string;
  /** Accent: primary buttons, focus rings, active states. */
  accent: string;
  /** Text on an accent fill. */
  onAccent: string;
  success: string;
  destructive: string;
  border: string;
}

/**
 * Build a theme whose palette is fully specified rather than derived: the
 * midground stays the body text colour, while the accent drives `primary`
 * and `ring` through `colorOverrides`. Used by the palette themes below.
 */
function paletteTheme(
  base: Pick<DashboardTheme, "name" | "label" | "description">,
  t: PaletteTokens,
  extra: Partial<DashboardTheme> = {},
): DashboardTheme {
  const [r, g, b] = parseColor(t.accent) ?? [255, 255, 255];
  return {
    ...base,
    palette: {
      background: { hex: t.bg, alpha: 1 },
      midground: { hex: t.ink, alpha: 1 },
      foreground: { hex: "#ffffff", alpha: 0 },
      warmGlow: `rgba(${r}, ${g}, ${b}, 0.3)`,
      noiseOpacity: 0,
    },
    typography: DEFAULT_TYPOGRAPHY,
    layout: { ...DEFAULT_LAYOUT, radius: "0.75rem" },
    colorOverrides: {
      card: t.panel,
      cardForeground: t.ink,
      popover: t.panel,
      popoverForeground: t.ink,
      primary: t.accent,
      primaryForeground: t.onAccent,
      secondary: t.panelSoft,
      secondaryForeground: t.ink,
      muted: t.panelSoft,
      mutedForeground: t.muted,
      accent: t.panelSoft,
      accentForeground: t.ink,
      destructive: t.destructive,
      success: t.success,
      border: t.border,
      input: t.border,
      ring: t.accent,
    },
    seriesColors: { inputTokenAccent: t.muted, outputTokenAccent: t.accent },
    swatchColors: [t.bg, t.accent, t.ink],
    terminalBackground: t.bg,
    terminalForeground: t.ink,
    ...extra,
  };
}

const SERIF_STACK = '"Iowan Old Style", "Palatino Linotype", Georgia, serif';

/** Quiet charcoal with sage accents. */
export const obsidianTheme = paletteTheme(
  { name: "obsidian", label: "Obsidian", description: "Quiet charcoal with sage accents" },
  { bg: "#131716", panel: "#181e1c", panelSoft: "#222b27", ink: "#e8efeb", muted: "#a0aea6",
    accent: "#91baa5", onAccent: "#131716", success: "#99cdb2", destructive: "#f19e98", border: "#35443c" },
);

/** Light mode — warm paper, clear type and deep forest green. */
export const porcelainTheme = paletteTheme(
  { name: "porcelain", label: "Porcelain", description: "Light mode — warm paper and deep forest green" },
  { bg: "#f4f3ef", panel: "#ffffff", panelSoft: "#e9eee9", ink: "#25352e", muted: "#637269",
    accent: "#286453", onAccent: "#ffffff", success: "#2e7157", destructive: "#b13c36", border: "#cbd5cb" },
  { terminalBackground: "#fbfaf7" },
);

/** Amber phosphor on black, set in monospace — instrument-panel feel. */
export const amberTheme = paletteTheme(
  { name: "amber", label: "Amber", description: "Amber phosphor on black, set in monospace" },
  { bg: "#080907", panel: "#10110c", panelSoft: "#16170f", ink: "#e9e3d4", muted: "#a39c89",
    accent: "#ffc233", onAccent: "#171b16", success: "#9fd69a", destructive: "#ff7a6b", border: "#3a3522" },
  {
    typography: { ...DEFAULT_TYPOGRAPHY, fontSans: SYSTEM_MONO, letterSpacing: "0.01em" },
    layout: { ...DEFAULT_LAYOUT, radius: "0.125rem" },
    customCSS: "h1, h2, h3 { letter-spacing: 0.04em; text-transform: uppercase; }",
  },
);

/** Gold on deep burgundy with a calligraphic display face. */
export const regaliaTheme = paletteTheme(
  { name: "regalia", label: "Regalia", description: "Gold on deep burgundy with calligraphic headings" },
  { bg: "#3a0f1a", panel: "#4a1622", panelSoft: "#5a1e2c", ink: "#ecd9a0", muted: "#c9a86f",
    accent: "#e6c34d", onAccent: "#2a0a13", success: "#b8c07a", destructive: "#e8a06a", border: "#6a2434" },
  {
    typography: {
      ...DEFAULT_TYPOGRAPHY,
      fontSans: `"IM Fell English", ${SERIF_STACK}`,
      fontDisplay: `"Great Vibes", "Snell Roundhand", "Segoe Script", cursive`,
      fontUrl: "https://fonts.googleapis.com/css2?family=Great+Vibes&family=IM+Fell+English:ital@0;1&display=swap",
      baseSize: "16px",
    },
    // IM Fell sets old-style figures; tables and counts read better in a lining serif.
    customCSS: `h1, h2 { font-family: var(--theme-font-display); font-weight: 400; letter-spacing: 0; line-height: 1.25; }
h1 { font-size: 2.6em; }
h2 { font-size: 1.9em; }
td, th, time { font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif; }`,
  },
);

/** Light mode — warm paper and terracotta in a refined serif. */
export const sandstoneTheme = paletteTheme(
  { name: "sandstone", label: "Sandstone", description: "Light mode — warm paper and terracotta, serif headings" },
  { bg: "#f6f1e7", panel: "#fffdf8", panelSoft: "#efe7d6", ink: "#3a2c1d", muted: "#7c6a54",
    accent: "#a8642a", onAccent: "#ffffff", success: "#5c7a4a", destructive: "#b1503a", border: "#d8ccb4" },
  {
    typography: {
      ...DEFAULT_TYPOGRAPHY,
      fontDisplay: `"Fraunces", ${SERIF_STACK}`,
      fontUrl: "https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400&display=swap",
    },
    terminalBackground: "#fffdf8",
    customCSS: "h1, h2, h3 { font-family: var(--theme-font-display); letter-spacing: -0.02em; }",
  },
);

/** Deep violet with cyan signals. */
export const nebulaTheme = paletteTheme(
  { name: "nebula", label: "Nebula", description: "Deep violet with cyan signals" },
  { bg: "#0d0b1a", panel: "#171334", panelSoft: "#211b47", ink: "#e9e6fb", muted: "#a9a3d0",
    accent: "#8b7bf0", onAccent: "#14112b", success: "#7fd6c0", destructive: "#f0a4c4", border: "#332c5e" },
  { seriesColors: { inputTokenAccent: "#5fd0e0", outputTokenAccent: "#8b7bf0" } },
);

/** Palette themes, exported for the legibility test. */
export const PALETTE_THEMES: DashboardTheme[] = [
  obsidianTheme, porcelainTheme, amberTheme, regaliaTheme, sandstoneTheme, nebulaTheme,
];

export const BUILTIN_THEMES: Record<string, DashboardTheme> = {
  default: defaultTheme,
  "default-large": defaultLargeTheme,
  "nous-blue": nousBlueTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  cyberpunk: cyberpunkTheme,
  rose: roseTheme,
  obsidian: obsidianTheme,
  porcelain: porcelainTheme,
  amber: amberTheme,
  regalia: regaliaTheme,
  sandstone: sandstoneTheme,
  nebula: nebulaTheme,
};
