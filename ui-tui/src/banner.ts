import type { ThemeColors } from './theme.js'

const RICH_RE = /\[(?:bold\s+)?(?:dim\s+)?(#(?:[0-9a-fA-F]{3,8}))\]([\s\S]*?)(\[\/\])/g

export function parseRichMarkup(markup: string): Line[] {
  const lines: Line[] = []

  for (const raw of markup.split('\n')) {
    const trimmed = raw.trimEnd()

    if (!trimmed) {
      lines.push(['', ' '])

      continue
    }

    const matches = [...trimmed.matchAll(RICH_RE)]

    if (!matches.length) {
      lines.push(['', trimmed])

      continue
    }

    let cursor = 0

    for (const m of matches) {
      const before = trimmed.slice(cursor, m.index)

      if (before) {
        lines.push(['', before])
      }

      lines.push([m[1]!, m[2]!])
      cursor = m.index! + m[0].length
    }

    if (cursor < trimmed.length) {
      lines.push(['', trimmed.slice(cursor)])
    }
  }

  return lines
}

const LOGO_ART = [
  '██╗  ██╗ █████╗ ██████╗ ███████╗███████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗',
  '██║  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝',
  '███████║███████║██║  ██║█████╗  ███████╗    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ',
  '██╔══██║██╔══██║██║  ██║██╔══╝  ╚════██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ',
  '██║  ██║██║  ██║██████╔╝███████╗███████║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ',
  '╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   '
]

const CADUCEUS_ART = [
  '       ▫    ▫       ',
  '       ▪    ▪       ',
  '      ▪▫    ▫▪      ',
  '     ▪▪      ▪▫     ',
  '    ▫▪▪      ▪▪▫    ',
  '    ▪▪▫      ▫▪▪    ',
  '   ▪▪▪       ▫▪▪▪   ',
  '  ▪▪▪▪        ▪▪▪▫  ',
  ' ▫▪▪▪▫        ▪▪▪▪▫ ',
  ' ▪▪▪▪▫        ▫▪▪▪▪ ',
  '▪▪▪▪▪▫        ▫▪▪▪▪▪',
  '▪▪▪▪▪▪▪▫    ▫▪▪▪▪▪▪▪',
  '  ▫▪▪▪▪▫  ▫▪▪▪▪▪▪▫  ',
  '    ▫▫  ▫▪▪▪▪▪▪▫    ',
  '     ▫▪▪▪▪▪▪▪▫      ',
  '   ▫▪▪▪▪▪▪▪▫  ▫▪▫   ',
  ' ▫▪▪▪▪▪▪▪▫  ▫▪▪▪▪▪▫ ',
  '▪▪▪▪▪▪▪▪    ▪▪▪▪▪▪▪▪',
  ' ▫▪▪▪▪▪▪▪  ▪▪▪▪▪▪▪▫ ',
  '    ▪▪▪▪▪▪▪▪▪▪▪▫    ',
  '      ▫▪▪▪▪▪▪▫      ',
  '        ▫▪▪▫        '
]

// Garnet / dark-red gradient (top → bottom), independent of the active theme
// so the Hades caduceus always renders in its signature deep reds.
const CADUCEUS_COLORS = [
  '#A31621',
  '#A31621',
  '#9B111E',
  '#9B111E',
  '#8B0000',
  '#8B0000',
  '#8B0000',
  '#7B1113',
  '#7B1113',
  '#7B1113',
  '#800020',
  '#800020',
  '#800020',
  '#6E0D0D',
  '#6E0D0D',
  '#6E0D0D',
  '#5C0011',
  '#5C0011',
  '#5C0011',
  '#4A0E0E',
  '#4A0E0E',
  '#4A0E0E'
] as const

const LOGO_GRADIENT = [0, 0, 1, 1, 2, 2] as const

const colorize = (art: string[], gradient: readonly number[], c: ThemeColors): Line[] => {
  const p = [c.primary, c.accent, c.border, c.muted]

  return art.map((text, i) => [p[gradient[i]!] ?? c.muted, text])
}

export const LOGO_WIDTH = Math.max(...LOGO_ART.map(line => line.length))
export const CADUCEUS_WIDTH = Math.max(...CADUCEUS_ART.map(line => line.length))

export const logo = (c: ThemeColors, customLogo?: string): Line[] =>
  customLogo ? parseRichMarkup(customLogo) : colorize(LOGO_ART, LOGO_GRADIENT, c)

export const caduceus = (_c: ThemeColors, customHero?: string): Line[] =>
  customHero ? parseRichMarkup(customHero) : CADUCEUS_ART.map((text, i) => [CADUCEUS_COLORS[i] ?? '#4A0E0E', text])

export const artWidth = (lines: Line[]) => lines.reduce((m, [, t]) => Math.max(m, t.length), 0)

type Line = [string, string]
