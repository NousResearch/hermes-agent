export interface ThemeColors {
  gold: string
  amber: string
  bronze: string
  cornsilk: string
  dim: string

  label: string
  ok: string
  error: string
  warn: string

  statusBg: string
  statusFg: string
  statusGood: string
  statusWarn: string
  statusBad: string
  statusCritical: string
}

export interface ThemeBrand {
  name: string
  icon: string
  prompt: string
  welcome: string
  goodbye: string
  tool: string
}

export interface Theme {
  color: ThemeColors
  brand: ThemeBrand
}

export const DEFAULT_THEME: Theme = {
  color: {
    gold: '#FFD700',
    amber: '#FFBF00',
    bronze: '#CD7F32',
    cornsilk: '#FFF8DC',
    dim: '#B8860B',

    label: '#4dd0e1',
    ok: '#4caf50',
    error: '#ef5350',
    warn: '#ffa726',

    statusBg: '#1a1a2e',
    statusFg: '#C0C0C0',
    statusGood: '#8FBC8F',
    statusWarn: '#FFD700',
    statusBad: '#FF8C00',
    statusCritical: '#FF6B6B'
  },

  brand: {
    name: 'Hermes Agent',
    icon: '⚕',
    prompt: '❯',
    welcome: 'Type your message or /help for commands.',
    goodbye: 'Goodbye! ⚕',
    tool: '┊'
  }
}

export function fromSkin(colors: Record<string, string>, branding: Record<string, string>): Theme {
  const d = DEFAULT_THEME
  const c = (k: string) => colors[k]

  return {
    color: {
      gold: c('banner_title') ?? d.color.gold,
      amber: c('banner_accent') ?? d.color.amber,
      bronze: c('banner_border') ?? d.color.bronze,
      cornsilk: c('banner_text') ?? d.color.cornsilk,
      dim: c('banner_dim') ?? d.color.dim,

      label: c('ui_label') ?? d.color.label,
      ok: c('ui_ok') ?? d.color.ok,
      error: c('ui_error') ?? d.color.error,
      warn: c('ui_warn') ?? d.color.warn,

      statusBg: d.color.statusBg,
      statusFg: d.color.statusFg,
      statusGood: c('ui_ok') ?? d.color.statusGood,
      statusWarn: c('ui_warn') ?? d.color.statusWarn,
      statusBad: d.color.statusBad,
      statusCritical: d.color.statusCritical
    },

    brand: {
      name: branding.agent_name ?? d.brand.name,
      icon: d.brand.icon,
      prompt: branding.prompt_symbol ?? d.brand.prompt,
      welcome: branding.welcome ?? d.brand.welcome,
      goodbye: branding.goodbye ?? d.brand.goodbye,
      tool: d.brand.tool
    }
  }
}
