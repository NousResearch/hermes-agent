import { INTERPOLATION_RE, LONG_MSG } from '../constants.js'

export const compactPreview = (s: string, max: number) => {
  const one = s.replace(/\s+/g, ' ').trim()

  return !one ? '' : one.length > max ? one.slice(0, max - 1) + '…' : one
}

export const estimateRows = (text: string, w: number) =>
  text.split('\n').reduce((s, l) => s + Math.max(1, Math.ceil(Math.max(1, l.length) / w)), 0)

export const flat = (r: Record<string, string[]>) => Object.values(r).flat()

export const fmtK = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`)

export const hasInterpolation = (s: string) => {
  INTERPOLATION_RE.lastIndex = 0

  return INTERPOLATION_RE.test(s)
}

export const pick = <T>(a: T[]) => a[Math.floor(Math.random() * a.length)]!

export const userDisplay = (text: string): string => {
  if (text.length <= LONG_MSG) {
    return text
  }

  const first = text.split('\n')[0]?.trim() ?? ''
  const words = first.split(/\s+/).filter(Boolean)
  const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80)

  return `${prefix || '(message)'} [long message]`
}
