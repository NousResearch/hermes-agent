import { contrastRatio, ensureContrast, mix, parseColor, relativeLuminance, toHex } from '@hermes/shared/color'
import type { SkinBranding, SkinColors } from '@hermes/shared/skin'

import { desaturate, grayOf, liftForContrast } from './lib/color.js'

export interface ThemeColors {
  primary: string
  accent: string
  border: string
  text: string
  muted: string
  completionBg: string
  completionCurrentBg: string
  completionMetaBg: string
  completionMetaCurrentBg: string

  label: string
  ok: string
  error: string
  warn: string

  /** Tool-call markers (● bullet, tool spinner). Defaults to `accent`. */
  tool: string
  /** Reasoning/thinking body text. Defaults to `muted`. */
  thinking: string

  /** Code-block syntax highlight. Default to accent/text/border/muted. */
  syntaxString: string
  syntaxNumber: string
  syntaxKeyword: string
  syntaxComment: string

  prompt: string
  sessionLabel: string
  sessionBorder: string

  statusBg: string
  statusFg: string
  statusGood: string
  statusWarn: string
  statusBad: string
  statusCritical: string
  selectionBg: string

  diffAdded: string
  diffRemoved: string
  diffAddedWord: string
  diffRemovedWord: string

  shellDollar: string
}

export interface ThemeBrand {
  name: string
  icon: string
  prompt: string
  welcome: string
  goodbye: string
  tool: string
  helpHeader: string
}

export interface Theme {
  color: ThemeColors
  brand: ThemeBrand
  bannerLogo: string
  bannerHero: string
  glyphs: ThemeGlyphs
}

// ── Color math ───────────────────────────────────────────────────────
//
// Generic color computation lives in @hermes/shared/color (the primitives,
// shared with the desktop) and lib/color.ts (TUI-only lifts and re-toning);
// this file keeps only the ANSI-256 remapping that is specific to the
// limited-palette Apple Terminal path. contrastRatio/ensureContrast are
// re-exported for existing consumers (tests, /theme-info).

export { contrastRatio, ensureContrast }

const XTERM_6_LEVELS = [0, 95, 135, 175, 215, 255] as const
const ANSI_LIGHT_MAX_LUMINANCE = 0.72
const ANSI_LIGHT_TARGET_LUMINANCE = 0.34
const ANSI_LIGHT_MIN_SATURATION = 0.22
const ANSI_MUTED_BUCKET = 245

const ANSI_NORMALIZED_FOREGROUNDS: readonly (keyof ThemeColors)[] = [
  'text',
  'label',
  'ok',
  'error',
  'warn',
  'prompt',
  'statusFg',
  'statusGood',
  'statusWarn',
  'statusBad',
  'statusCritical',
  'shellDollar',
  'tool'
]

const ANSI_MUTED_FOREGROUNDS: readonly (keyof ThemeColors)[] = ['muted', 'sessionLabel', 'sessionBorder', 'thinking']

function xtermEightBitRgb(colorNumber: number): [number, number, number] {
  if (colorNumber >= 232) {
    const value = 8 + (colorNumber - 232) * 10

    return [value, value, value]
  }

  if (colorNumber >= 16) {
    const offset = colorNumber - 16

    return [
      XTERM_6_LEVELS[Math.floor(offset / 36) % 6]!,
      XTERM_6_LEVELS[Math.floor(offset / 6) % 6]!,
      XTERM_6_LEVELS[offset % 6]!
    ]
  }

  return [0, 0, 0]
}

const rgbLuminance = (red: number, green: number, blue: number): number =>
  relativeLuminance(toHex([red, green, blue])) ?? 0

function rgbToHsl(red: number, green: number, blue: number): [number, number, number] {
  const rn = red / 255
  const gn = green / 255
  const bn = blue / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const lightness = (max + min) / 2

  if (max === min) {
    return [0, 0, lightness]
  }

  const delta = max - min
  const saturation = lightness > 0.5 ? delta / (2 - max - min) : delta / (max + min)

  const hue =
    max === rn ? (gn - bn) / delta + (gn < bn ? 6 : 0) : max === gn ? (bn - rn) / delta + 2 : (rn - gn) / delta + 4

  return [hue / 6, saturation, lightness]
}

function circularDistance(a: number, b: number): number {
  const distance = Math.abs(a - b)

  return Math.min(distance, 1 - distance)
}

// Mirrors @hermes/ink's colorize.ts. Keep local: app code compiles from
// ui-tui/src, while @hermes/ink is bundled separately from packages/.
function richEightBitColorNumber(red: number, green: number, blue: number): number {
  const [, saturation, lightness] = rgbToHsl(red, green, blue)

  if (saturation < 0.15) {
    const gray = Math.round(lightness * 25)

    return gray === 0 ? 16 : gray === 25 ? 231 : 231 + gray
  }

  const sixRed = red < 95 ? red / 95 : 1 + (red - 95) / 40
  const sixGreen = green < 95 ? green / 95 : 1 + (green - 95) / 40
  const sixBlue = blue < 95 ? blue / 95 : 1 + (blue - 95) / 40

  return 16 + 36 * Math.round(sixRed) + 6 * Math.round(sixGreen) + Math.round(sixBlue)
}

function bestReadableAnsiColor(red: number, green: number, blue: number): number {
  const [hue, saturation, lightness] = rgbToHsl(red, green, blue)
  let bestColor = richEightBitColorNumber(red, green, blue)
  let bestScore = Number.POSITIVE_INFINITY

  for (let colorNumber = 16; colorNumber <= 255; colorNumber += 1) {
    const [candidateRed, candidateGreen, candidateBlue] = xtermEightBitRgb(colorNumber)
    const candidateLuminance = rgbLuminance(candidateRed, candidateGreen, candidateBlue)

    if (candidateLuminance > ANSI_LIGHT_MAX_LUMINANCE) {
      continue
    }

    const [candidateHue, candidateSaturation, candidateLightness] = rgbToHsl(
      candidateRed,
      candidateGreen,
      candidateBlue
    )

    const saturationFloorPenalty =
      candidateSaturation < ANSI_LIGHT_MIN_SATURATION ? (ANSI_LIGHT_MIN_SATURATION - candidateSaturation) * 3 : 0

    const score =
      circularDistance(candidateHue, hue) * 4 +
      Math.abs(candidateSaturation - Math.max(ANSI_LIGHT_MIN_SATURATION, saturation)) * 0.8 +
      Math.abs(candidateLightness - Math.min(lightness, ANSI_LIGHT_TARGET_LUMINANCE)) * 2 +
      saturationFloorPenalty

    if (score < bestScore) {
      bestColor = colorNumber
      bestScore = score
    }
  }

  return bestColor
}

function normalizeAnsiForeground(color: string): string {
  const rgb = parseColor(color)

  if (!rgb) {
    return color
  }

  const richAnsi = richEightBitColorNumber(rgb[0], rgb[1], rgb[2])
  const richRgb = xtermEightBitRgb(richAnsi)

  const ansi =
    rgbLuminance(richRgb[0], richRgb[1], richRgb[2]) > ANSI_LIGHT_MAX_LUMINANCE
      ? bestReadableAnsiColor(rgb[0], rgb[1], rgb[2])
      : richAnsi

  return `ansi256(${ansi})`
}

const ANSI256_RE = /^ansi256\((\d{1,3})\)$/

/**
 * The literal `#rrggbb` a theme tone paints as, or '' when it has none.
 *
 * The inverse of `normalizeAnsiForeground`: tones are not uniformly hex, since
 * a limited-palette terminal rewrites the foregrounds to `ansi256(N)`.
 * Consumers needing a real color — OSC 10/11, which only speak `#rrggbb` —
 * resolve through here rather than hex-testing the tone, which would silently
 * skip exactly the terminals that did the quantizing.
 */
export function themeToneHex(tone: string): string {
  const ansi = ANSI256_RE.exec(tone.trim())

  if (ansi) {
    const n = Number(ansi[1])

    return n <= 255 ? toHex(xtermEightBitRgb(n)) : ''
  }

  const rgb = parseColor(tone)

  return rgb ? toHex(rgb) : ''
}

// ── Defaults ─────────────────────────────────────────────────────────

const BRAND: ThemeBrand = {
  name: 'Hermes Agent',
  icon: '☤',
  prompt: '❯',
  welcome: 'Type your message or /help for commands.',
  goodbye: 'Goodbye! ☤',
  tool: '┊',
  helpHeader: '(^_^)? Commands'
}

const cleanPromptSymbol = (s: string | undefined, fallback: string) => {
  const cleaned = String(s ?? '')
    .replace(/\s+/g, ' ')
    .trim()

  return cleaned || fallback
}

// ── Chrome glyphs (display.tui_glyph_preset) ─────────────────────────
//
// Every glyph the TUI draws as CHROME — status icons, tree rails, spinners,
// ok/fail marks, separators — resolves through one of three tiers instead of
// being authored per call site, so a terminal whose font lacks the classic
// Hermes glyphs shows a readable TUI instead of tofu boxes:
//
//   nerd    — Nerd Font icons (`\uF00C` etc.) for the icon slots. Patched
//             fonts keep the standard box-drawing/braille ranges, so rails,
//             separators and spinners stay identical to `unicode`.
//   unicode — the classic Hermes glyphs. THE DEFAULT, byte-identical to the
//             pre-preset rendering: no existing setup changes.
//   ascii   — pure 7-bit (`[ok]`, `|-`, `|`), the floor for terminals with no
//             font control at all (embedded consoles, some SSH clients).
//
// Width contract: the spinner frames are one column in every tier (the
// status-bar budget reserves exactly that), and the rails/separators/
// chevrons are one column in `unicode`/`nerd`. The ok/fail/warn MARKS are
// wider in the ascii tier (`[ok]` vs `✓`), so any width arithmetic around a
// mark must measure the preset value (`stringWidth(t.glyphs.ok)`) instead of
// a literal.
export const GLYPH_PRESETS = ['ascii', 'nerd', 'unicode'] as const

export type GlyphPreset = (typeof GLYPH_PRESETS)[number]

/** The tier every terminal we have no signal about can render. */
export const DEFAULT_GLYPH_PRESET: GlyphPreset = 'unicode'

/** Lifecycle vocabulary of the agent/subagent surfaces (see lib/subagentGlyph). */
export type GlyphStatus =
  | 'cancelled'
  | 'completed'
  | 'dispatched'
  | 'error'
  | 'failed'
  | 'finalizing'
  | 'interrupted'
  | 'queued'
  | 'rejected'
  | 'running'
  | 'timeout'
  | 'unknown'

export interface ThemeGlyphs {
  /** Empty context-bar cell. */
  barEmpty: string
  /** Filled context-bar cell. */
  barFilled: string
  /** Resume-when-idle marker. */
  back: string
  /** Battery read-out marker. */
  battery: string
  /** Charging marker (battery read-out). */
  bolt: string
  /** Running-tool / list bullet. */
  bullet: string
  /** Cache-hit percentage marker. */
  cache: string
  /** Streaming caret at the end of live text. */
  caret: string
  /** Background-subagent chain marker. */
  chain: string
  /** Collapsed-section chevron. */
  chevronClosed: string
  /** Expanded-section chevron. */
  chevronOpen: string
  /** Latency clock. */
  clock: string
  /** Signed delta marker (dev credits). */
  delta: string
  /** Inline separator dot (`· `). */
  dot: string
  /** Timeline/event marker. */
  event: string
  /** Failed / negative mark. */
  fail: string
  /** Focus-view badge. */
  focus: string
  /** Assistant-row gutter mark (chrome stand-in for the brand's `┊`). */
  gutter: string
  /** Brand mark shown beside the agent name. */
  icon: string
  /** Completed / positive mark. */
  ok: string
  /** Paused marker. */
  pause: string
  /** Input-prompt mark for the user gutter. */
  prompt: string
  /** Tree rail, last child (`└─ `). */
  railLast: string
  /** Tree rail, middle child (`├─ `). */
  railMid: string
  /** Tree rail, vertical pass-through (`│ `). */
  railPipe: string
  /** Horizontal rule / status-bar lead (`─ `). */
  rule: string
  /** Scrollbar thumb cell. */
  scrollbar: string
  /** Status-segment separator (` │ `). */
  sep: string
  /** Chrome spinner cycle (one column per frame). */
  spinner: readonly string[]
  /** Per-status icon. */
  status: Record<GlyphStatus, string>
  /** Tool-role gutter glyph. */
  tool: string
  /** Throughput marker (tokens/sec). */
  up: string
  /** Warning mark. */
  warn: string
}

// The `unicode` tier's spinner is the braille cycle the busy indicator has
// always used (`unicode-animations`' `braille` spinner, 80 ms/frame).
const BRAILLE_SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'] as const

const CLASSIC_STATUS: Record<GlyphStatus, string> = {
  cancelled: '■',
  completed: '✓',
  dispatched: '○',
  error: '⚠',
  failed: '✗',
  finalizing: '◐',
  interrupted: '■',
  queued: '○',
  rejected: '⊘',
  running: '●',
  timeout: '⌛',
  unknown: '·'
}

const UNICODE_GLYPHS: ThemeGlyphs = {
  barEmpty: '░',
  barFilled: '█',
  back: '↩',
  battery: '🔋',
  bolt: '⚡',
  bullet: '●',
  cache: '◎',
  caret: '▍',
  chain: '⛓',
  chevronClosed: '▸',
  chevronOpen: '▾',
  clock: '◷',
  delta: 'Δ',
  dot: '·',
  event: '◈',
  fail: '✗',
  focus: '◉',
  gutter: '┊',
  icon: '☤',
  ok: '✓',
  pause: '⏸',
  prompt: '❯',
  railLast: '└─',
  railMid: '├─',
  railPipe: '│',
  rule: '─',
  scrollbar: '┃',
  sep: '│',
  spinner: BRAILLE_SPINNER,
  status: CLASSIC_STATUS,
  tool: '⚡',
  up: '↑',
  warn: '⚠'
}

// Icon slots only: Nerd Fonts ship every standard box-drawing/braille glyph
// unpatched, so the rails, separators, rule, scrollbar and spinner stay on
// the `unicode` values instead of inventing PUA equivalents for them.
// Codepoints follow the reference table (oh-my-pi `symbols.ts` NERD map):
// nf-fa-check F00C, nf-fa-times F00D, nf-fa-exclamation_triangle F12A,
// nf-fa-circle F111 / circle_o F10C, nf-fa-spinner F110, nf-fa-square
// F04D, nf-fa-ban F05E, nf-fa-hourglass F254, nf-fa-bolt F0E7,
// nf-fa-caret_right/caret_down F0DA/F0D7, nf-fa-battery_full F240,
// nf-fa-question F128.
const NERD_GLYPHS: ThemeGlyphs = {
  ...UNICODE_GLYPHS,
  battery: '\uF240',
  bolt: '\uF0E7',
  chevronClosed: '\uF0DA',
  chevronOpen: '\uF0D7',
  fail: '\uF00D',
  ok: '\uF00C',
  tool: '\uF0E7',
  warn: '\uF12A',
  status: {
    cancelled: '\uF04D',
    completed: '\uF00C',
    dispatched: '\uF10C',
    error: '\uF12A',
    failed: '\uF00D',
    finalizing: '\uF111',
    interrupted: '\uF04D',
    queued: '\uF10C',
    rejected: '\uF05E',
    running: '\uF110',
    timeout: '\uF254',
    unknown: '\uF128'
  }
}

// `[ok]` and friends take the width the classic marks take in the transcript
// header slot; the rails keep the classic `tree --charset=ascii` shapes so a
// depth level still reads as a level.
const ASCII_GLYPHS: ThemeGlyphs = {
  barEmpty: '-',
  barFilled: '#',
  back: '<-',
  battery: 'B',
  bolt: '!',
  bullet: '*',
  cache: '%',
  caret: '|',
  chain: '@',
  chevronClosed: '>',
  chevronOpen: 'v',
  clock: '~',
  delta: 'D',
  dot: '.',
  event: '+',
  fail: '[x]',
  focus: '*',
  gutter: '|',
  icon: '',
  ok: '[ok]',
  pause: '||',
  prompt: '>',
  railLast: '`-',
  railMid: '|-',
  railPipe: '|',
  rule: '-',
  scrollbar: '#',
  sep: '|',
  spinner: ['|', '/', '-', '\\'],
  status: {
    cancelled: '[-]',
    completed: '[ok]',
    dispatched: '[ ]',
    error: '[!]',
    failed: '[x]',
    finalizing: '[~]',
    interrupted: '[-]',
    queued: '[ ]',
    rejected: '[/]',
    running: '[~]',
    timeout: '[t]',
    unknown: '.'
  },
  tool: '*',
  up: '^',
  warn: '[!]'
}

export const GLYPH_TABLES: Record<GlyphPreset, ThemeGlyphs> = {
  ascii: ASCII_GLYPHS,
  nerd: NERD_GLYPHS,
  unicode: UNICODE_GLYPHS
}

const GLYPH_PRESET_SET: ReadonlySet<string> = new Set(GLYPH_PRESETS)

/** Normalize a configured value (`display.tui_glyph_preset`, possibly absent
 *  or hand-edited) to a shipped tier. Unknown/missing falls back to the
 *  default tier rather than throwing — a typo must not cost the user their
 *  chrome. */
export function normalizeGlyphPreset(raw: unknown): GlyphPreset {
  const value = typeof raw === 'string' ? raw.trim().toLowerCase() : ''

  return GLYPH_PRESET_SET.has(value) ? (value as GlyphPreset) : DEFAULT_GLYPH_PRESET
}

/** The glyph table for a configured value. */
export function glyphTable(raw: unknown): ThemeGlyphs {
  return GLYPH_TABLES[normalizeGlyphPreset(raw)]
}

// The active tier for themes BUILT from here on (skin arrival, background
// re-detection). Config sync sets it; every later theme resolution carries it,
// so a preset change survives a skin swap instead of reverting on the next
// theme rebuild.
let activeGlyphPreset: GlyphPreset = DEFAULT_GLYPH_PRESET

export const getGlyphPreset = (): GlyphPreset => activeGlyphPreset

export function setGlyphPreset(raw: unknown): GlyphPreset {
  activeGlyphPreset = normalizeGlyphPreset(raw)

  return activeGlyphPreset
}

/** Re-derive an already-resolved theme with a new tier. The theme object is
 *  the TUI's single render input, so swapping it re-renders every chrome
 *  consumer (and persists through the same boot-theme path skins use). */
export function themeWithGlyphPreset(theme: Theme, raw: unknown): Theme {
  const glyphs = glyphTable(raw)

  return {
    ...theme,
    brand: {
      ...theme.brand,
      icon: tierDefault(theme.brand.icon, 'icon', glyphs.icon),
      prompt: tierDefault(theme.brand.prompt, 'prompt', glyphs.prompt),
      tool: tierDefault(theme.brand.tool, 'gutter', glyphs.gutter)
    },
    glyphs
  }
}

/** The new tier's value for a brand mark, unless the current one was AUTHORED
 *  (a skin's `tool_prefix`, a user's `prompt_symbol`) — those are identity,
 *  and a preset change must not overwrite them. */
function tierDefault(current: string, key: 'gutter' | 'icon' | 'prompt', next: string): string {
  return GLYPH_PRESETS.some(preset => GLYPH_TABLES[preset][key] === current) ? next : current
}

/** Frames a chrome spinner should cycle. A tier whose own cycle is ASCII
 *  (the ascii floor) never borrows the caller's art — the decorative
 *  thinking/tool animations are braille and unreadable without a
 *  braille-capable font, which is exactly the case the tier exists for. */
export function spinnerFrames(glyphs: ThemeGlyphs, art: readonly string[] = []): readonly string[] {
  if (art.length && (glyphs.spinner[0]?.codePointAt(0) ?? 0) >= 0x7f) {
    return art
  }

  return glyphs.spinner
}

// ── Seeds → palette ──────────────────────────────────────────────────
//
// A palette is BUILT, not enumerated: skins/base themes supply identity seeds
// (text, primary, accent, semantic hues) and every secondary tone is derived
// by the mix ladder against the background. This is the desktop token system
// (seeds → color-mix → tokens) in terminal form — "dim" is definitionally a
// derivative of the theme's own base colors and can never be incoherent.

export interface ThemeSeeds {
  accent: string
  /** Identity fill override: active list-row chip (derived when omitted). */
  activeRow?: string
  bg: string
  border?: string
  error: string
  /** Identity tone override: muted/dim text (derived when omitted). */
  muted?: string
  ok: string
  primary: string
  prompt?: string
  /** Identity fill override: text-selection highlight (derived when omitted). */
  selection?: string
  shellDollar: string
  statusBad: string
  statusCritical: string
  statusGood: string
  statusWarn: string
  /** Identity fill override: panel/status surface (derived when omitted). */
  surface?: string
  text: string
  warn: string
}

const DIFF_DARK = {
  diffAdded: 'rgb(220,255,220)',
  diffRemoved: 'rgb(255,220,220)',
  diffAddedWord: 'rgb(36,138,61)',
  diffRemovedWord: 'rgb(207,34,46)'
}

const DIFF_LIGHT = {
  diffAdded: 'rgb(200,240,200)',
  diffRemoved: 'rgb(240,200,200)',
  diffAddedWord: 'rgb(27,94,32)',
  diffRemovedWord: 'rgb(183,28,28)'
}

export function buildPalette(seeds: ThemeSeeds, isLight: boolean): ThemeColors {
  const tones = deriveTones(seeds)
  const surface = seeds.surface ?? tones.surface
  const activeRow = seeds.activeRow ?? (seeds.surface ? mix(surface, seeds.accent, 0.22) : tones.activeRow)
  const muted = seeds.muted ?? tones.muted

  return {
    primary: seeds.primary,
    accent: seeds.accent,
    border: seeds.border ?? tones.border,
    text: seeds.text,
    muted,
    completionBg: surface,
    completionCurrentBg: activeRow,
    completionMetaBg: surface,
    completionMetaCurrentBg: activeRow,

    label: tones.label,
    ok: seeds.ok,
    error: seeds.error,
    warn: seeds.warn,

    // Element tokens: independently settable, but default to their semantic
    // parents (tool marker → accent, reasoning body → muted).
    tool: seeds.accent,
    thinking: muted,

    // Code-syntax tokens default to brand tokens (unchanged highlighting)
    // but are independently skinnable.
    syntaxString: seeds.accent,
    syntaxNumber: seeds.text,
    syntaxKeyword: seeds.border ?? tones.border,
    syntaxComment: muted,

    prompt: seeds.prompt ?? seeds.text,
    // sessionLabel/sessionBorder track the muted tone — "same role, same
    // colour" by design (#11300).
    sessionLabel: muted,
    sessionBorder: muted,

    statusBg: surface,
    statusFg: tones.statusFg,
    statusGood: seeds.statusGood,
    statusWarn: seeds.statusWarn,
    statusBad: seeds.statusBad,
    statusCritical: seeds.statusCritical,
    selectionBg: seeds.selection ?? tones.selection,

    ...(isLight ? DIFF_LIGHT : DIFF_DARK),
    shellDollar: seeds.shellDollar
  }
}

export const DARK_SEEDS: ThemeSeeds = {
  accent: '#FFBF00',
  // The classic Hermes navy surfaces are IDENTITY, not derivation drift —
  // keep them as explicit fill seeds (the ladder derives them for skins
  // that don't care).
  activeRow: '#333355',
  bg: '#101014',
  border: '#CD7F32',
  error: '#ef5350',
  ok: '#4caf50',
  primary: '#FFD700',
  prompt: '#FFF8DC',
  selection: '#3a3a55',
  shellDollar: '#4dabf7',
  statusBad: '#FF8C00',
  statusCritical: '#FF6B6B',
  statusGood: '#8FBC8F',
  statusWarn: '#FFD700',
  surface: '#1a1a2e',
  text: '#FFF8DC',
  warn: '#ffa726'
}

// Light-terminal seeds: darker golds/ambers that stay legible on white.
// The classic light-mode Hermes look was never hand-authored: for years the
// TUI emitted the DARK golds and hosts with xterm's minimumContrastRatio
// (Cursor defaults to 4.5) lifted them against white — hue and saturation
// kept, luminance clamped. These seeds are those exact lifts
// (liftForContrast(dark, '#ffffff', 4.5)), so hosts WITHOUT a contrast pass
// render the same thing Cursor always showed. Text/prompt stay ink — body
// copy historically rendered in the terminal's default near-black fg.
export const LIGHT_SEEDS: ThemeSeeds = {
  accent: '#956E00',
  bg: '#ffffff',
  border: '#A56628',
  error: '#C14240',
  ok: '#367E39',
  primary: '#867000',
  prompt: '#2B2014',
  shellDollar: '#377BB3',
  statusBad: '#A65A00',
  statusCritical: '#B94D4D',
  statusGood: '#5C7A5C',
  statusWarn: '#867000',
  text: '#3D2F13',
  warn: '#956115'
}

/**
 * Brand CHROME follows the glyph tier: `BRAND` owns the identity (name,
 * welcome/goodbye prose) while the marks the TUI draws in a gutter or prompt
 * slot come from the tier, so the ascii floor never paints `❯`/`┊`.
 * An authored `prompt_symbol` / skin `tool_prefix` still wins (see fromSkin) —
 * a user's own symbol is their choice, not chrome.
 */
const tierBrand = (brand: ThemeBrand, glyphs: ThemeGlyphs): ThemeBrand => ({
  ...brand,
  icon: glyphs.icon,
  prompt: glyphs.prompt,
  tool: glyphs.gutter
})

const baseTheme = (seeds: ThemeSeeds, isLight: boolean): Theme => {
  const glyphs = glyphTable(activeGlyphPreset)

  return {
    color: buildPalette(seeds, isLight),
    brand: tierBrand(BRAND, glyphs),
    bannerLogo: '',
    bannerHero: '',
    glyphs
  }
}

export const DARK_THEME: Theme = baseTheme(DARK_SEEDS, false)

export const LIGHT_THEME: Theme = baseTheme(LIGHT_SEEDS, true)

// ── Background-aware readability adaptation ─────────────────────────
//
// Mirrors the desktop app's theme contract (apps/desktop/src/themes): skins
// contribute accent IDENTITY; readability against the actual background is
// the theme engine's job, enforced in one place. Two guards, in the desktop's
// vocabulary:
//
//   * `ensureContrast` — foreground-role colors are step-mixed toward the
//     readable pole (black on light, white on dark) until they clear a
//     minimum WCAG contrast ratio against the real (or assumed) background.
//     Hue survives; washout doesn't.
//   * Fill polarity — background-role colors (completion menu, status bar,
//     selection) must match the terminal's polarity. Unlike the desktop, the
//     TUI does not own its canvas — panels sit directly on the terminal's
//     background — so a wrong-polarity fill (navy menu on a white terminal)
//     falls back to the base palette even when the skin authored it.

// Display shim — the "rendering gotcha" layer, calibrated against the look
// the maintainers standardized on (pixel-sampled from the reference
// screenshot): the beloved cross-polarity rendering is the AUTHORED palette
// displayed RAW — slate's ~1.5:1 pastels on white read as deliberate airy
// hierarchy, not a bug. So the floors are barely-visible rescues only:
//   * DISPLAY 1.45 sits just above slate-pastel territory (#c9d1d9 = 1.54,
//     passes raw, byte-identical) but just below true invisibility
//     (default's cream #FFF8DC = 1.08, gets rescued).
//   * SEMANTIC 2.2 for alert colors (ok/error/warn/status) — they carry
//     meaning and must never vanish.
// The lift itself is xterm.js's own multiplicative algorithm
// (liftForContrast), so on hosts that run their own minimumContrastRatio the
// two adjustments agree instead of fighting.
// Foreground floors are polarity-aware. On a DARK background the authored
// palette is already bright, so a modest floor only rescues the rare dark
// tone. On a LIGHT background — which in practice means a TRANSPARENT Cursor/
// terminal window compositing over a light editor, where xterm applies NO
// contrast lift of its own (there is no solid bg to measure against) — the
// beloved classic look is the authored palette rendered essentially RAW:
// vivid #FFD700 gold (~1.36:1), not a WCAG-darkened mustard. So the light
// floor is a near-invisible rescue only (catches cream #FFF8DC at 1.08 but
// leaves the golds untouched). Pixel-sampled target: #F5C242 (L61 S90),
// which the previous 1.45 floor crushed to #867000 (L26) — the reported mud.
const DISPLAY_MIN_CONTRAST = 1.45
const SEMANTIC_MIN_CONTRAST = 2.2
const LIGHT_DISPLAY_MIN_CONTRAST = 1.18
const LIGHT_SEMANTIC_MIN_CONTRAST = 1.6

const DISPLAY_FOREGROUNDS: readonly (keyof ThemeColors)[] = [
  'primary',
  'accent',
  'text',
  'label',
  'prompt',
  'statusFg',
  'border',
  'muted',
  'sessionLabel',
  'sessionBorder',
  'shellDollar'
]

const SEMANTIC_FOREGROUNDS: readonly (keyof ThemeColors)[] = [
  'ok',
  'error',
  'warn',
  'statusGood',
  'statusWarn',
  'statusBad',
  'statusCritical'
]

const ADAPTIVE_BACKGROUNDS: readonly (keyof ThemeColors)[] = [
  'completionBg',
  'completionCurrentBg',
  'completionMetaBg',
  'completionMetaCurrentBg',
  'statusBg',
  'selectionBg'
]

// Fill polarity limits: on light terminals a fill must stay light, and vice
// versa — there is no readable middle for a panel fill on the wrong pole.
const LIGHT_BG_MIN_LUMINANCE = 0.4
const DARK_BG_MAX_LUMINANCE = 0.35

function adaptColorsToBackground(colors: ThemeColors, isLight: boolean, base: ThemeColors, bg: string): ThemeColors {
  const out = { ...colors }
  const displayFloor = isLight ? LIGHT_DISPLAY_MIN_CONTRAST : DISPLAY_MIN_CONTRAST
  const semanticFloor = isLight ? LIGHT_SEMANTIC_MIN_CONTRAST : SEMANTIC_MIN_CONTRAST

  for (const key of DISPLAY_FOREGROUNDS) {
    out[key] = liftForContrast(out[key], bg, displayFloor)
  }

  for (const key of SEMANTIC_FOREGROUNDS) {
    out[key] = liftForContrast(out[key], bg, semanticFloor)
  }

  for (const key of ADAPTIVE_BACKGROUNDS) {
    const luminance = relativeLuminance(out[key])

    if (luminance === null) {
      continue
    }

    if (isLight ? luminance < LIGHT_BG_MIN_LUMINANCE : luminance > DARK_BG_MAX_LUMINANCE) {
      out[key] = base[key]
    }
  }

  return out
}

/** The background hex adaptation measures contrast against: the OSC-11
 *  answer when known (cached in HERMES_TUI_BACKGROUND), else the mode's
 *  assumed pole. */
function referenceBackground(isLight: boolean, env: NodeJS.ProcessEnv = process.env): string {
  const cached = (env.HERMES_TUI_BACKGROUND ?? '').trim()

  if (cached && backgroundLuminance(cached) !== null) {
    return cached.startsWith('#') ? cached : `#${cached}`
  }

  return isLight ? '#ffffff' : '#101014'
}

// ── Derived tone ladder (the desktop color-mix system) ──────────────
//
// A theme is a handful of SEEDS (text, primary, accent, border, status hues);
// every secondary tone — muted text, labels, surfaces, selection chips — is a
// color-mix derivative of those seeds against the real terminal background,
// exactly like the desktop's `--theme-*` seeds → `--ui-*` color-mix ladder in
// apps/desktop/src/styles.css. Skins therefore cannot ship an incoherent
// "dim": if they don't author a tone, it is DERIVED from their own identity,
// never inherited from another skin's palette.
//
// Mix knobs are the single source of truth for tone hierarchy. (The classic
// prompt_toolkit CLI still reads the built-in skins' complete authored
// palettes; those were generated with the same math.)

export interface ThemeTones {
  /** Secondary/dim text: receded accent (dark) / primary-ink blend (light). */
  muted: string
  /** Field labels: one step brighter than muted, same family. */
  label: string
  /** Status-bar default text: the gray of slightly-receded text. */
  statusFg: string
  /** Raised panel fill: background nudged toward the (softened) accent. */
  surface: string
  /** Active list-row chip: surface tinted with accent. */
  activeRow: string
  /** Text-selection highlight: bg tinted with the theme's blue (light) or accent (dark). */
  selection: string
  /** Border fallback: accent receded toward the background. */
  border: string
}

/**
 * The fitted tone ladder. Knobs are REVERSE-ENGINEERED from the original
 * hand-tuned palettes (grid-search over mix/desaturate formula families
 * against the pre-refactor literals + every authored skin palette; see the
 * "reproduces the original hand-tuned tones" test for the contract):
 *
 *   dark muted  #CC9B1F ≈ desaturate(mix(accent, bg, .19), .16)  (err 3)
 *   dark label  #DAA520 ≈ desaturate(mix(accent, bg, .13), .16)  (err 3)
 *   dark status #C0C0C0 = grayOf(mix(text, bg, .24))             (err 0)
 *   light muted #946C08 ≈ desaturate(accent, .05)                (err 2)
 *   light label #8E6B13 ≈ desaturate(mix(accent, text, .03), .15) (err 2)
 *   light status #6F6F6F = grayOf(mix(text, bg, .30))            (err 1)
 *   light surface #F5F5F5 ≈ bg + softened accent                 (err 5)
 *   light chip  #E0D1BF = mix(surface, accent, .25)              (err 8)
 *   light selection #D4E4F7 ≈ mix(bg, shellDollar, .20)          (err 7)
 *
 * The light targets are the LIFT CANON: liftForContrast(dark literal,
 * white, 4.5) — what xterm's minimumContrastRatio showed on light hosts
 * for years — not hand-picked browns (those read as desaturated mud).
 *
 * The classic dark navy fills (#1a1a2e/#333355/#3a3a55) are IRREDUCIBLE from
 * gold seeds — the search bottoms out at gray, err 10–17 — so they remain
 * explicit identity seeds on DARK_SEEDS rather than pretending to be math.
 */
export function deriveTones(seeds: {
  accent: string
  bg: string
  primary: string
  shellDollar?: string
  text: string
}): ThemeTones {
  const { accent, bg, text } = seeds
  const isLight = (relativeLuminance(bg) ?? 0) > 0.5
  // Fill tint keeps most of the accent's chroma — a heavier desaturate here
  // read as washed-out ("a little too desat") next to authored fills.
  const surface = mix(bg, desaturate(accent, 0.15), isLight ? 0.045 : 0.09)

  return {
    // Light knobs are fitted to the lift canon (xterm minimumContrastRatio
    // 4.5 of the classic dark golds against white — see LIGHT_SEEDS), not
    // to ink blends: muted #946C08 ≈ desat(accent .05), label #8E6B13 ≈
    // desat(mix(accent, text, .03), .15), statusFg #6F6F6F ≈ gray 30% lift.
    muted: isLight ? desaturate(accent, 0.05) : desaturate(mix(accent, bg, 0.19), 0.16),
    label: isLight ? desaturate(mix(accent, text, 0.03), 0.15) : desaturate(mix(accent, bg, 0.13), 0.16),
    statusFg: grayOf(mix(text, bg, isLight ? 0.3 : 0.24)),
    surface,
    activeRow: mix(surface, accent, 0.25),
    selection: isLight && seeds.shellDollar ? mix(bg, seeds.shellDollar, 0.2) : mix(surface, accent, 0.28),
    border: mix(accent, bg, 0.25)
  }
}

const TRUE_RE = /^(?:1|true|yes|on)$/
const FALSE_RE = /^(?:0|false|no|off)$/

// TERM_PROGRAM fallback allow-list for terminals whose default profile is
// light and which may not expose COLORFGBG. This currently includes Apple
// Terminal. Explicit HERMES_TUI_THEME / COLORFGBG signals above still win,
// so dark Apple Terminal profiles that advertise a dark background stay dark.
const LIGHT_DEFAULT_TERM_PROGRAMS = new Set<string>(['Apple_Terminal'])

// Best-effort RGB → luminance check.  Currently only accepts a 3- or
// 6-digit hex value (with or without a leading `#`); the env var name
// `HERMES_TUI_BACKGROUND` is intentionally generic so a future OSC11
// query helper can cache its answer there too, but additional formats
// (rgb()/hsl()/named colours) would need explicit parsing here first.
const LUMA_LIGHT_THRESHOLD = 0.6

// Strict allow-list: parseInt(..., 16) silently truncates at the first
// non-hex character (e.g. `fffgff` would parse as `fff` and yield a
// false-positive "white" reading), so reject anything that doesn't match
// the canonical 3- or 6-digit shape up front.
const HEX_3_RE = /^[0-9a-f]{3}$/
const HEX_6_RE = /^[0-9a-f]{6}$/

function backgroundLuminance(raw: string): null | number {
  const v = raw.trim().toLowerCase()

  if (!v) {
    return null
  }

  const hex = v.startsWith('#') ? v.slice(1) : v

  const rgb = HEX_6_RE.test(hex)
    ? [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)]
    : HEX_3_RE.test(hex)
      ? [parseInt(hex[0]! + hex[0]!, 16), parseInt(hex[1]! + hex[1]!, 16), parseInt(hex[2]! + hex[2]!, 16)]
      : null

  if (!rgb) {
    return null
  }

  // Rec. 709 luma — close enough for "is this background bright".
  return (0.2126 * rgb[0]! + 0.7152 * rgb[1]! + 0.0722 * rgb[2]!) / 255
}

// Pick light vs dark with ordered, explainable signals (#11300):
//
//   1. `HERMES_TUI_LIGHT` boolean — `1`/`true`/`yes`/`on` → light;
//      `0`/`false`/`no`/`off` → dark.  Either explicit value wins
//      regardless of any later signal.
//   2. `HERMES_TUI_THEME` named override — `light` / `dark` win over
//      every signal below.
//   3. `HERMES_TUI_BACKGROUND` hex hint (3- or 6-digit) — luminance
//      ≥ LUMA_LIGHT_THRESHOLD → light.
//   4. `COLORFGBG` last field — XFCE / rxvt / Terminal.app emit
//      slot 7 or 15 on light profiles; 0–15 ranges are otherwise
//      treated as authoritatively dark so the TERM_PROGRAM
//      allow-list below cannot override an explicit dark profile.
//   5. `TERM_PROGRAM` light-default allow-list.
//
// Anything we can't decide stays dark — the default Hermes palette
// is the dark one.
export function detectLightMode(
  env: NodeJS.ProcessEnv = process.env,
  // Injectable so tests can prove the COLORFGBG-over-TERM_PROGRAM
  // precedence rule even though the production allow-list is empty.
  lightDefaultTermPrograms: ReadonlySet<string> = LIGHT_DEFAULT_TERM_PROGRAMS
): boolean {
  const lightFlag = (env.HERMES_TUI_LIGHT ?? '').trim().toLowerCase()

  if (TRUE_RE.test(lightFlag)) {
    return true
  }

  if (FALSE_RE.test(lightFlag)) {
    return false
  }

  const themeFlag = (env.HERMES_TUI_THEME ?? '').trim().toLowerCase()

  if (themeFlag === 'light') {
    return true
  }

  if (themeFlag === 'dark') {
    return false
  }

  const bgHint = backgroundLuminance(env.HERMES_TUI_BACKGROUND ?? '')

  if (bgHint !== null) {
    return bgHint >= LUMA_LIGHT_THRESHOLD
  }

  const colorfgbg = (env.COLORFGBG ?? '').trim()

  if (colorfgbg) {
    // Validate as a decimal integer before coercing — `Number('')` is 0,
    // so a malformed `COLORFGBG='15;'` would otherwise look like an
    // authoritative dark slot and incorrectly block the TERM_PROGRAM
    // allow-list.  Anything that isn't pure digits falls through.
    const lastField = colorfgbg.split(';').at(-1) ?? ''

    if (/^\d+$/.test(lastField)) {
      const bg = Number(lastField)

      if (bg === 7 || bg === 15) {
        return true
      }

      // Slots 0–6 and 8–14 are the dark half of the 0–15 ANSI range.
      // When COLORFGBG is set we trust it as authoritative — a non-light
      // value here shouldn't get overridden by the TERM_PROGRAM allow-list.
      if (bg >= 0 && bg < 16) {
        return false
      }
    }
  }

  const termProgram = (env.TERM_PROGRAM ?? '').trim()

  return lightDefaultTermPrograms.has(termProgram)
}

function shouldNormalizeAnsiLightTheme(env: NodeJS.ProcessEnv = process.env, isLight = detectLightMode(env)): boolean {
  const colorTerm = (env.COLORTERM ?? '').trim().toLowerCase()
  const termProgram = (env.TERM_PROGRAM ?? '').trim()

  return termProgram === 'Apple_Terminal' && colorTerm !== 'truecolor' && colorTerm !== '24bit' && isLight
}

export function normalizeThemeForAnsiLightTerminal(
  theme: Theme,
  env: NodeJS.ProcessEnv = process.env,
  isLight = detectLightMode(env)
): Theme {
  if (!shouldNormalizeAnsiLightTheme(env, isLight)) {
    return theme
  }

  const color = { ...theme.color }

  for (const key of ANSI_NORMALIZED_FOREGROUNDS) {
    color[key] = normalizeAnsiForeground(color[key])
  }

  for (const key of ANSI_MUTED_FOREGROUNDS) {
    color[key] = `ansi256(${ANSI_MUTED_BUCKET})`
  }

  return { ...theme, color }
}

const DEFAULT_LIGHT_MODE = detectLightMode()

export const DEFAULT_THEME: Theme = normalizeThemeForAnsiLightTerminal(
  DEFAULT_LIGHT_MODE ? LIGHT_THEME : DARK_THEME,
  process.env,
  DEFAULT_LIGHT_MODE
)

/**
 * The skinless theme for the CURRENT light-mode signals. Unlike the frozen
 * module-load DEFAULT_THEME, this re-reads the environment — so it picks up
 * the OSC-11 background answer cached into HERMES_TUI_BACKGROUND after
 * startup. Used when the terminal background arrives before (or without) a
 * gateway skin.
 */
export function defaultThemeForCurrentBackground(env: NodeJS.ProcessEnv = process.env): Theme {
  const isLight = detectLightMode(env)

  return normalizeThemeForAnsiLightTerminal(isLight ? LIGHT_THEME : DARK_THEME, env, isLight)
}

// ── Skin → Theme ─────────────────────────────────────────────────────

/** The skin's authored canvas as a #-prefixed hex, or null when absent/junk. */
const authoredBackground = (raw: string | undefined): null | string => {
  const v = (raw ?? '').trim()

  return backgroundLuminance(v) === null ? null : v.startsWith('#') ? v : `#${v}`
}

/**
 * A skin that authors a background OWNS its polarity: the TUI paints the
 * terminal with it (applySkin → OSC-11), so contrast adaptation, the derived
 * tone ladder, and ANSI light-terminal normalization must all run against the
 * skin's canvas — not the host profile the skin just covered. (A pure-black
 * skin on a light Apple Terminal otherwise gets its text remapped for a light
 * background it painted over: invisible.) Host detection still governs skins
 * without a background — they render on the terminal's own surface.
 */
export function skinIsLight(colors: SkinColors, env: NodeJS.ProcessEnv = process.env): boolean {
  const authored = backgroundLuminance(colors['background'] ?? '')

  return authored === null ? detectLightMode(env) : authored >= LUMA_LIGHT_THRESHOLD
}

export function fromSkin(
  colors: SkinColors,
  branding: SkinBranding,
  bannerLogo = '',
  bannerHero = '',
  toolPrefix = '',
  helpHeader = ''
): Theme {
  // Polarity: the skin's own canvas when it authors one (see skinIsLight);
  // otherwise live host detection (not the module-load snapshot — by the time
  // the gateway skin arrives, the OSC-11 probe has usually answered and cached
  // itself into HERMES_TUI_BACKGROUND. See #applySkin / syncThemeToTerminalBackground).
  const skinBg = authoredBackground(colors['background'])
  const isLight = skinIsLight(colors)
  const bg = skinBg ?? referenceBackground(isLight)
  const base = isLight ? LIGHT_SEEDS : DARK_SEEDS
  const d = isLight ? LIGHT_THEME : DARK_THEME
  // The active tier, re-read per call: a skin arriving AFTER a preset change
  // (or after the config sync) must not revert the chrome.
  const skinGlyphs = glyphTable(activeGlyphPreset)
  const c = (k: string) => colors[k]

  const hasSkinColors = Object.keys(colors).length > 0

  // 1. Seeds: the skin's identity. Anything it doesn't define comes from the
  //    base seeds for this polarity. The base's IDENTITY FILLS (Hermes navy
  //    surfaces, gold muted) only carry over for the skinless default — a
  //    skin with its own identity derives its fills from its own seeds.
  const identityFills: Partial<ThemeSeeds> = hasSkinColors
    ? {}
    : { activeRow: base.activeRow, muted: base.muted, selection: base.selection, surface: base.surface }

  const seeds: ThemeSeeds = {
    ...identityFills,
    accent: c('ui_accent') ?? c('banner_accent') ?? base.accent,
    bg,
    border: c('ui_border') ?? c('banner_border') ?? base.border,
    error: c('ui_error') ?? base.error,
    ok: c('ui_ok') ?? base.ok,
    primary: c('ui_primary') ?? c('banner_title') ?? base.primary,
    prompt: c('prompt') ?? c('banner_text') ?? base.prompt,
    shellDollar: c('shell_dollar') ?? base.shellDollar,
    statusBad: c('status_bar_bad') ?? base.statusBad,
    statusCritical: c('status_bar_critical') ?? c('ui_error') ?? base.statusCritical,
    statusGood: c('status_bar_good') ?? c('ui_ok') ?? base.statusGood,
    statusWarn: c('status_bar_warn') ?? c('ui_warn') ?? base.statusWarn,
    text: c('ui_text') ?? c('banner_text') ?? base.text,
    warn: c('ui_warn') ?? base.warn
  }

  // 2. Derive: every secondary tone is a color-mix of the seeds against the
  //    REAL background — dim is a derivative of the skin's own identity.
  const derived = buildPalette(seeds, isLight)

  // 3. Authored tone overrides: a skin may still hand-tune any tone; the
  //    derived ladder is the default, not a cage. Chip/selection re-derive
  //    from the FINAL surface so dependents stay coherent with overrides.
  // `background` is theme-sdk's cross-surface base: the TUI paints the
  // terminal with it (applySkin → setTerminalBackground), so panels/status
  // must sit on it too — fall the surface back to it below completion_menu_bg.
  const surface = c('completion_menu_bg') ?? c('background') ?? derived.completionBg

  // Re-mix the chip only when the skin authored its own surface; otherwise
  // the derived value already carries the identity seeds (e.g. Hermes navy).
  const activeRow =
    c('completion_menu_current_bg') ??
    (c('completion_menu_bg') ? mix(surface, seeds.accent, 0.22) : derived.completionCurrentBg)

  const assembled: ThemeColors = {
    ...derived,
    muted: c('banner_dim') ?? derived.muted,
    label: c('ui_label') ?? derived.label,
    completionBg: surface,
    completionCurrentBg: activeRow,
    completionMetaBg: c('completion_menu_meta_bg') ?? surface,
    completionMetaCurrentBg: c('completion_menu_meta_current_bg') ?? activeRow,
    sessionLabel: c('session_label') ?? c('banner_dim') ?? derived.sessionLabel,
    sessionBorder: c('session_border') ?? c('banner_dim') ?? derived.sessionBorder,
    statusBg: c('status_bar_bg') ?? surface,
    statusFg: c('status_bar_text') ?? c('ui_text') ?? c('banner_text') ?? derived.statusFg,
    selectionBg: c('selection_bg') ?? c('completion_menu_current_bg') ?? derived.selectionBg,
    // Element tokens + skinnable diffs (theme-sdk): overridable, else the
    // derived defaults (tool→accent, thinking→muted, diff_* → DIFF_* ladder).
    // thinking tracks the EFFECTIVE muted (banner_dim override included), not
    // just derived.muted, so recoloring muted carries the reasoning body with it.
    tool: c('ui_tool') ?? derived.tool,
    thinking: c('ui_thinking') ?? c('banner_dim') ?? derived.thinking,
    diffAdded: c('diff_added') ?? derived.diffAdded,
    diffRemoved: c('diff_removed') ?? derived.diffRemoved,
    diffAddedWord: c('diff_added_word') ?? derived.diffAddedWord,
    diffRemovedWord: c('diff_removed_word') ?? derived.diffRemovedWord,
    // Code-syntax tokens: overridable, else the derived brand-token defaults.
    syntaxString: c('syntax_string') ?? derived.syntaxString,
    syntaxNumber: c('syntax_number') ?? derived.syntaxNumber,
    syntaxKeyword: c('syntax_keyword') ?? derived.syntaxKeyword,
    syntaxComment: c('syntax_comment') ?? c('banner_dim') ?? derived.syntaxComment
  }

  // 4. Guard: contrast floors against the real background + fill polarity.
  //    Wrong-polarity fills fall back to the DERIVED value, which is
  //    polarity-correct by construction (mixed from the background itself).
  //    ANSI-limited light Apple Terminal keeps its bespoke ansi256
  //    normalization (below) instead.
  const adapted = shouldNormalizeAnsiLightTheme(process.env, isLight)
    ? assembled
    : adaptColorsToBackground(assembled, isLight, derived, bg)

  return normalizeThemeForAnsiLightTerminal(
    {
      // The element tokens theme-sdk introduced (ui_primary, ui_text,
      // ui_border, ui_ok/warn/error, ui_tool, ui_thinking, shell_dollar,
      // status_bar_*, diff_*) are read into `seeds`/`assembled` above and
      // flow through buildPalette → adaptColorsToBackground, so `adapted`
      // already honors them AND applies #20379's contrast/polarity machinery.
      // Emitting a hand-mapped color block here would bypass that adaptation
      // and regress theme quality.
      color: adapted,

      brand: {
        name: branding.agent_name ?? d.brand.name,
        icon: skinGlyphs.icon,
        prompt: cleanPromptSymbol(branding.prompt_symbol, skinGlyphs.prompt),
        welcome: branding.welcome ?? d.brand.welcome,
        goodbye: branding.goodbye ?? d.brand.goodbye,
        tool: toolPrefix || skinGlyphs.gutter,
        helpHeader: branding.help_header ?? (helpHeader || d.brand.helpHeader)
      },

      bannerLogo,
      bannerHero,

      glyphs: skinGlyphs
    },
    process.env,
    isLight
  )
}
