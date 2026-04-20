export interface ThemeColors {
  gold: string
  amber: string
  bronze: string
  cornsilk: string
  dim: string
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

export const DARK_THEME: Theme = {
  color: {
    gold: '#FFD700',
    amber: '#FFBF00',
    bronze: '#CD7F32',
    cornsilk: '#FFF8DC',
    dim: '#B8860B',
    completionBg: '#FFFFFF',
    completionCurrentBg: mix('#FFFFFF', '#FFBF00', 0.25),

    label: '#DAA520',
    ok: '#4caf50',
    error: '#ef5350',
    warn: '#ffa726',

    prompt: '#FFF8DC',
    sessionLabel: '#B8860B',
    sessionBorder: '#B8860B',

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
    gold: '#8B6914',
    amber: '#A0651C',
    bronze: '#7A4F1F',
    cornsilk: '#3D2F13',
    dim: '#7A5A0F',
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

// Pick light vs dark. Explicit `HERMES_TUI_LIGHT` wins; otherwise sniff
// `COLORFGBG` (set by XFCE Terminal, rxvt, Terminal.app, etc.) — last field is the
// background ANSI index; 7/15 are the "white" slots most light themes emit (#11300).
export function detectLightMode(env: NodeJS.ProcessEnv = process.env): boolean {
  const explicit = (env.HERMES_TUI_LIGHT ?? '').trim().toLowerCase()

  if (/^(?:1|true|yes|on)$/.test(explicit)) {
    return true
  }

  if (/^(?:0|false|no|off)$/.test(explicit)) {
    return false
  }

  const bg = Number((env.COLORFGBG ?? '').trim().split(';').at(-1))

  return bg === 7 || bg === 15
}

export const DEFAULT_THEME: Theme = detectLightMode() ? LIGHT_THEME : DARK_THEME

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

  const amber = c('ui_accent') ?? c('banner_accent') ?? d.color.amber
  const accent = c('banner_accent') ?? c('banner_title') ?? d.color.amber
  const dim = c('banner_dim') ?? d.color.dim

  return {
    color: {
      gold: c('banner_title') ?? d.color.gold,
      amber,
      bronze: c('banner_border') ?? d.color.bronze,
      cornsilk: c('banner_text') ?? d.color.cornsilk,
      dim,
      completionBg: c('completion_menu_bg') ?? '#FFFFFF',
      completionCurrentBg: c('completion_menu_current_bg') ?? mix('#FFFFFF', accent, 0.25),

      label: c('ui_label') ?? d.color.label,
      ok: c('ui_ok') ?? d.color.ok,
      error: c('ui_error') ?? d.color.error,
      warn: c('ui_warn') ?? d.color.warn,

      prompt: c('prompt') ?? c('banner_text') ?? d.color.prompt,
      sessionLabel: c('session_label') ?? dim,
      sessionBorder: c('session_border') ?? dim,

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
      prompt: branding.prompt_symbol ?? d.brand.prompt,
      welcome: branding.welcome ?? d.brand.welcome,
      goodbye: branding.goodbye ?? d.brand.goodbye,
      tool: toolPrefix || d.brand.tool,
      helpHeader: branding.help_header ?? (helpHeader || d.brand.helpHeader)
    },

    bannerLogo,
    bannerHero
  }
}
