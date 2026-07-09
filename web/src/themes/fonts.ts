/**
 * Curated UI-font catalog for the dashboard font override.
 *
 * The font override is an independent layer that sits ON TOP of the active
 * theme: a theme still ships its own `typography.fontSans` default, but a
 * user can pick any font here and it persists across theme switches. Picking
 * "Theme default" clears the override and returns to whatever the active
 * theme specifies.
 *
 * Why a curated catalog instead of a free-text font name + URL box: the
 * `fontUrl` is injected into the page as a `<link rel="stylesheet">`, so
 * accepting an arbitrary user-supplied URL would be a self-XSS / SSRF-ish
 * footgun in the dashboard. A vetted catalog keeps the injected origins
 * fixed (system stacks + Google Fonts) while still giving real choice. The
 * matching allow-list on the backend (`_FONT_CHOICES` in web_server.py)
 * rejects any id not defined here.
 *
 * Keep `FONT_CHOICES` in sync with `_FONT_CHOICES` in
 * `hermes_cli/web_server.py` — the ids must match exactly.
 */

/** System stacks reused from presets so "System" choices need no webfont. */
const SYSTEM_SANS =
  'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const SYSTEM_MONO =
  'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace, "Noto Sans Mono"';
const SYSTEM_SERIF =
  'Georgia, Cambria, "Times New Roman", Times, serif';

export type FontCategory = "sans" | "serif" | "mono";

export interface FontChoice {
  /** Stable id persisted in config / localStorage. */
  id: string;
  /** Human-readable label shown in the picker. */
  label: string;
  /** Rough grouping for the picker. */
  category: FontCategory;
  /** CSS font-family stack applied to `--theme-font-sans` (+ display). */
  stack: string;
  /** Optional Google-Fonts (or other vetted) stylesheet URL. */
  fontUrl?: string;
}

/** Sentinel id meaning "no override — use the active theme's font". */
export const THEME_DEFAULT_FONT_ID = "theme";

const GF = (family: string): string =>
  `https://fonts.googleapis.com/css2?family=${family}&display=swap`;

/**
 * The curated set. Order is the display order in the picker (grouped by
 * category in the UI). `stack` always ends in a system fallback so a font
 * that fails to load still renders something sane.
 */
export const FONT_CHOICES: FontChoice[] = [
  // ── System (no webfont fetch) ──────────────────────────────────────────
  { id: "system-sans", label: "System Sans", category: "sans", stack: SYSTEM_SANS },
  { id: "system-serif", label: "System Serif", category: "serif", stack: SYSTEM_SERIF },
  { id: "system-mono", label: "System Mono", category: "mono", stack: SYSTEM_MONO },

  // ── Sans ────────────────────────────────────────────────────────────────
  {
    id: "inter",
    label: "Inter",
    category: "sans",
    stack: `"Inter", ${SYSTEM_SANS}`,
    fontUrl: GF("Inter:wght@400;500;600;700"),
  },
  {
    id: "ibm-plex-sans",
    label: "IBM Plex Sans",
    category: "sans",
    stack: `"IBM Plex Sans", ${SYSTEM_SANS}`,
    fontUrl: GF("IBM+Plex+Sans:wght@400;500;600;700"),
  },
  {
    id: "work-sans",
    label: "Work Sans",
    category: "sans",
    stack: `"Work Sans", ${SYSTEM_SANS}`,
    fontUrl: GF("Work+Sans:wght@400;500;600;700"),
  },
  {
    id: "atkinson-hyperlegible",
    label: "Atkinson Hyperlegible",
    category: "sans",
    stack: `"Atkinson Hyperlegible", ${SYSTEM_SANS}`,
    fontUrl: GF("Atkinson+Hyperlegible:wght@400;700"),
  },
  {
    id: "dm-sans",
    label: "DM Sans",
    category: "sans",
    stack: `"DM Sans", ${SYSTEM_SANS}`,
    fontUrl: GF("DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700"),
  },

  // ── Serif ─────────────────────────────────────────────────────────────
  {
    id: "spectral",
    label: "Spectral",
    category: "serif",
    stack: `"Spectral", ${SYSTEM_SERIF}`,
    fontUrl: GF("Spectral:wght@400;500;600;700"),
  },
  {
    id: "fraunces",
    label: "Fraunces",
    category: "serif",
    stack: `"Fraunces", ${SYSTEM_SERIF}`,
    fontUrl: GF("Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600"),
  },
  {
    id: "source-serif",
    label: "Source Serif 4",
    category: "serif",
    stack: `"Source Serif 4", ${SYSTEM_SERIF}`,
    fontUrl: GF("Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700"),
  },

  // ── Mono ──────────────────────────────────────────────────────────────
  {
    id: "jetbrains-mono",
    label: "JetBrains Mono",
    category: "mono",
    stack: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl: GF("JetBrains+Mono:wght@400;500;700"),
  },
  {
    id: "ibm-plex-mono",
    label: "IBM Plex Mono",
    category: "mono",
    stack: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl: GF("IBM+Plex+Mono:wght@400;500;700"),
  },
  {
    id: "space-mono",
    label: "Space Mono",
    category: "mono",
    stack: `"Space Mono", ${SYSTEM_MONO}`,
    fontUrl: GF("Space+Mono:wght@400;700"),
  },
];

const FONT_BY_ID: Record<string, FontChoice> = Object.fromEntries(
  FONT_CHOICES.map((f) => [f.id, f]),
);

/** Look up a font choice by id. Returns undefined for the theme-default
 *  sentinel and for any unknown id. */
export function getFontChoice(id: string | null | undefined): FontChoice | undefined {
  if (!id || id === THEME_DEFAULT_FONT_ID) return undefined;
  return FONT_BY_ID[id];
}

/** Whether an id refers to a real catalog font (vs. theme-default/unknown). */
export function isOverrideFont(id: string | null | undefined): boolean {
  return getFontChoice(id) !== undefined;
}
