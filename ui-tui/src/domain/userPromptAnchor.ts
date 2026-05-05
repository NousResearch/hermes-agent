import type { Theme } from '../theme.js'

export const USER_PROMPT_ANCHOR_PURPLE = '#B084FF'
export const USER_PROMPT_ANCHOR_TEXT = '#F2EAFE'

const VALID_BOX_STYLES = new Set(['round'])

export interface UserPromptAnchorStyle {
  borderColor: string
  borderStyle: 'round'
  marginBottom: number
  marginTop: number
  textColor: string
  title: string
  titleColor: string
}

export interface UserPromptAnchorConfig {
  accent_color?: unknown
  boxed?: unknown
  box_style?: unknown
  margin_bottom?: unknown
  margin_top?: unknown
  text_color?: unknown
}

const normalizeColor = (raw: unknown, fallback: string): string => {
  if (typeof raw !== 'string') {
    return fallback
  }

  const value = raw.trim()
  return value || fallback
}

const normalizeMargin = (raw: unknown, fallback: number): number => {
  const value = typeof raw === 'number' ? raw : typeof raw === 'string' ? Number.parseInt(raw, 10) : Number.NaN

  return Number.isFinite(value) ? Math.max(0, value) : fallback
}

export function userPromptAnchorStyle(
  _theme?: Theme,
  config: UserPromptAnchorConfig = {}
): UserPromptAnchorStyle | null {
  if (config.boxed === false) {
    return null
  }

  const requestedBoxStyle = typeof config.box_style === 'string' ? config.box_style.trim().toLowerCase() : 'round'
  const borderStyle = VALID_BOX_STYLES.has(requestedBoxStyle) ? 'round' : 'round'
  const borderColor = normalizeColor(config.accent_color, USER_PROMPT_ANCHOR_PURPLE)

  return {
    borderColor,
    borderStyle,
    marginBottom: normalizeMargin(config.margin_bottom, 2),
    marginTop: normalizeMargin(config.margin_top, 2),
    textColor: normalizeColor(config.text_color, USER_PROMPT_ANCHOR_TEXT),
    title: 'You',
    titleColor: borderColor
  }
}
