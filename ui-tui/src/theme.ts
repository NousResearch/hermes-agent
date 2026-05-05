export interface ThemeColors {
  primary: string
  accent: string
  border: string
  text: string
  muted: string
  completionBg: string
  completionCurrentBg: string

  label: string
  ok: string
  error: string
  warn: string

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
}

// ── Color math ───────────────────────────────────────────────────────

function parseHex(h: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(h)

  if (!m) {
    return null
  }

  const n = parseInt(m[1]!, 16)

  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
}

function mix(a: string, b: string, t: number) {
  const pa = parseHex(a)
  const pb = parseHex(b)

  if (!pa || !pb) {
    return a
  }

  const lerp = (i: 0 | 1 | 2) => Math.round(pa[i] + (pb[i] - pa[i]) * t)

  return '#' + ((1 << 24) | (lerp(0) << 16) | (lerp(1) << 8) | lerp(2)).toString(16).slice(1)
}

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
  'shellDollar'
]

const ANSI_MUTED_FOREGROUNDS: readonly (keyof ThemeColors)[] = ['muted', 'sessionLabel', 'sessionBorder']

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

function channelLuminance(value: number): number {
  const normalized = value / 255

  return normalized <= 0.03928 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4
}

function relativeLuminance(red: number, green: number, blue: number): number {
  return 0.2126 * channelLuminance(red) + 0.7152 * channelLuminance(green) + 0.0722 * channelLuminance(blue)
}

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
    max === rn
      ? (gn - bn) / delta + (gn < bn ? 6 : 0)
      : max === gn
        ? (bn - rn) / delta + 2
        : (rn - gn) / delta + 4

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
    const candidateLuminance = relativeLuminance(candidateRed, candidateGreen, candidateBlue)

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
  const rgb = parseHex(color)

  if (!rgb) {
    return color
  }

  const richAnsi = richEightBitColorNumber(rgb[0], rgb[1], rgb[2])
  const richRgb = xtermEightBitRgb(richAnsi)

  const ansi = relativeLuminance(richRgb[0], richRgb[1], richRgb[2]) > ANSI_LIGHT_MAX_LUMINANCE
    ? bestReadableAnsiColor(rgb[0], rgb[1], rgb[2])
    : richAnsi

  return `ansi256(${ansi})`
}

// ── Defaults ─────────────────────────────────────────────────────────

const BRAND: ThemeBrand = {
  name: 'Hermes Agent',
  icon: '⚕',
  prompt: '❯',
  welcome: 'Type your message or /help for commands.',
  goodbye: 'Goodbye! ⚕',
  tool: '┊',
  helpHeader: '(^_^)? Commands'
}

const cleanPromptSymbol = (s: string | undefined, fallback: string) => {
  const cleaned = String(s ?? '')
    .replace(/\s+/g, ' ')
    .trim()

  return cleaned || fallback
}

export const DARK_THEME: Theme = {
  color: {
    primary: '#FFD700',
    accent: '#FFBF00',
    border: '#CD7F32',
    text: '#FFF8DC',
    muted: '#CC9B1F',
    // Bumped from the old `#B8860B` darkgoldenrod (~53% luminance) which
    // read as barely-visible on dark terminals for long body text.  The
    // new value sits ~60% luminance — readable without losing the "muted /
    // secondary" semantic.  Field labels still use `label` (65%) which
    // stays brighter so hierarchy holds.
    completionBg: '#FFFFFF',
    completionCurrentBg: mix('#FFFFFF', '#FFBF00', 0.25),

    label: '#DAA520',
    ok: '#4caf50',
    error: '#ef5350',
    warn: '#ffa726',

    prompt: '#FFF8DC',
    // sessionLabel/sessionBorder intentionally track the `dim` value — they
    // are "same role, same colour" by design.  fromSkin's banner_dim fallback
    // relies on this pairing (#11300).
    sessionLabel: '#CC9B1F',
    sessionBorder: '#CC9B1F',

    statusBg: '#1a1a2e',
    statusFg: '#C0C0C0',
    statusGood: '#8FBC8F',
    statusWarn: '#FFD700',
    statusBad: '#FF8C00',
    statusCritical: '#FF6B6B',
    selectionBg: '#3a3a55',

    diffAdded: 'rgb(220,255,220)',
    diffRemoved: 'rgb(255,220,220)',
    diffAddedWord: 'rgb(36,138,61)',
    diffRemovedWord: 'rgb(207,34,46)',
    shellDollar: '#4dabf7'
  },

  brand: BRAND,

  bannerLogo: '',
  bannerHero: ''
}

// Light-terminal palette: darker golds/ambers that stay legible on white
// backgrounds. Same shape as DARK_THEME so `fromSkin` still layers on top
// cleanly (#11300).
export const LIGHT_THEME: Theme = {
  color: {
    primary: '#8B6914',
    accent: '#A0651C',
    border: '#7A4F1F',
    text: '#3D2F13',
    muted: '#7A5A0F',
    completionBg: '#F5F5F5',
    completionCurrentBg: mix('#F5F5F5', '#A0651C', 0.25),

    label: '#7A5A0F',
    ok: '#2E7D32',
    error: '#C62828',
    warn: '#E65100',

    prompt: '#2B2014',
    sessionLabel: '#7A5A0F',
    sessionBorder: '#7A5A0F',

    statusBg: '#F5F5F5',
    statusFg: '#333333',
    statusGood: '#2E7D32',
    statusWarn: '#8B6914',
    statusBad: '#D84315',
    statusCritical: '#B71C1C',
    selectionBg: '#D4E4F7',

    diffAdded: 'rgb(200,240,200)',
    diffRemoved: 'rgb(240,200,200)',
    diffAddedWord: 'rgb(27,94,32)',
    diffRemovedWord: 'rgb(183,28,28)',
    shellDollar: '#1565C0'
  },

  brand: BRAND,

  bannerLogo: '',
  bannerHero: ''
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

// ── Skin → Theme ─────────────────────────────────────────────────────

export function fromSkin(
  colors: Record<string, string>,
  branding: Record<string, string>,
  bannerLogo = '',
  bannerHero = '',
  toolPrefix = '',
  helpHeader = ''
): Theme {
  const d = DEFAULT_THEME
  const c = (k: string) => colors[k]

  const accent = c('ui_accent') ?? c('banner_accent') ?? d.color.accent
  const bannerAccent = c('banner_accent') ?? c('banner_title') ?? d.color.accent
  const muted = c('banner_dim') ?? d.color.muted
  const completionBg = c('completion_menu_bg') ?? d.color.completionBg

  return normalizeThemeForAnsiLightTerminal({
    color: {
      primary: c('ui_primary') ?? c('banner_title') ?? d.color.primary,
      accent,
      border: c('ui_border') ?? c('banner_border') ?? d.color.border,
      text: c('ui_text') ?? c('banner_text') ?? d.color.text,
      muted,
      completionBg,
      completionCurrentBg: c('completion_menu_current_bg') ?? mix(completionBg, bannerAccent, 0.25),

      label: c('ui_label') ?? d.color.label,
      ok: c('ui_ok') ?? d.color.ok,
      error: c('ui_error') ?? d.color.error,
      warn: c('ui_warn') ?? d.color.warn,

      prompt: c('prompt') ?? c('banner_text') ?? d.color.prompt,
      sessionLabel: c('session_label') ?? muted,
      sessionBorder: c('session_border') ?? muted,

      statusBg: d.color.statusBg,
      statusFg: d.color.statusFg,
      statusGood: c('ui_ok') ?? d.color.statusGood,
      statusWarn: c('ui_warn') ?? d.color.statusWarn,
      statusBad: d.color.statusBad,
      statusCritical: d.color.statusCritical,
      selectionBg: c('selection_bg') ?? d.color.selectionBg,

      diffAdded: d.color.diffAdded,
      diffRemoved: d.color.diffRemoved,
      diffAddedWord: d.color.diffAddedWord,
      diffRemovedWord: d.color.diffRemovedWord,
      shellDollar: c('shell_dollar') ?? d.color.shellDollar
    },

    brand: {
      name: branding.agent_name ?? d.brand.name,
      icon: d.brand.icon,
      prompt: cleanPromptSymbol(branding.prompt_symbol, d.brand.prompt),
      welcome: branding.welcome ?? d.brand.welcome,
      goodbye: branding.goodbye ?? d.brand.goodbye,
      tool: toolPrefix || d.brand.tool,
      helpHeader: branding.help_header ?? (helpHeader || d.brand.helpHeader)
    },

    bannerLogo,
    bannerHero
  }, process.env, DEFAULT_LIGHT_MODE)
}
