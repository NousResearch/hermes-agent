import type { ThemeColors } from './theme.js'

// Tags the old parser recognized: an optional bold/dim prefix plus a hex
// color, or a closing tag. Anything else ([foo], [#xyz], a bare [) is left
// as literal text.
const TAG_RE = /\[((?:bold\s+)?(?:dim\s+)?(#[0-9a-fA-F]{3,8}))\]|\[\/\]/g

/** One colored run inside a banner row: `[color, text]`. */
export type BannerSegment = [color: string, text: string]
/** One source line of banner art: the segments that share its row. */
export type BannerRow = BannerSegment[]

export function parseRichMarkup(markup: string): BannerRow[] {
  // One row per source line; every styled segment of that line stays on the
  // row as a nested span. The previous shape returned one entry PER SEGMENT,
  // so per-character gradient art (`[#a1]W[/][#b2]e[/]...`) exploded into one
  // terminal row per character. Tags nest (outer multi-line spans wrapping
  // per-character ones), so a stack tracks the open colors; `[/]` pops.
  // Spans may cross line boundaries — the color carries over instead of
  // leaking the raw tags into the output.
  const rows: BannerRow[] = []
  let row: BannerRow = []
  const stack: string[] = []

  const endRow = () => {
    // Mirror the old per-line trimEnd: drop a trailing whitespace-only tail.
    let end = row.length
    while (end > 0) {
      const seg = row[end - 1]!
      const trimmed = seg[1].replace(/\s+$/, '')
      if (trimmed.length === seg[1].length) break
      if (trimmed) {
        seg[1] = trimmed
        break
      }
      end -= 1
    }
    row.length = end
    rows.push(row.length ? row : [['', ' ']])
    row = []
  }

  const emit = (text: string) => {
    const color = stack[stack.length - 1] ?? ''
    const parts = text.split('\n')
    for (let i = 0; i < parts.length; i++) {
      if (i > 0) endRow()
      if (parts[i]) row.push([color, parts[i]!])
    }
  }

  // Fresh regex per call: TAG_RE is global and carries lastIndex state.
  const re = new RegExp(TAG_RE.source, 'g')
  let cursor = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(markup)) !== null) {
    emit(markup.slice(cursor, m.index))
    if (m[0] === '[/]') stack.pop()
    else stack.push(m[2]!)
    cursor = m.index + m[0].length
  }
  emit(markup.slice(cursor))
  // A trailing newline terminates a final (possibly blank) row, matching the
  // old split('\n') behavior; otherwise just flush the pending row.
  if (row.length > 0 || rows.length === 0 || markup.endsWith('\n')) endRow()

  return rows
}

const LOGO_ART = [
  '██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗',
  '██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝',
  '███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ',
  '██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ',
  '██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ',
  '╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   '
]

const CADUCEUS_ART = [
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀',
  '⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀',
  '⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀',
  '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀'
]

const LOGO_GRADIENT = [0, 0, 1, 1, 2, 2] as const
const CADUC_GRADIENT = [2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3] as const

const colorize = (art: string[], gradient: readonly number[], c: ThemeColors): BannerRow[] => {
  const p = [c.primary, c.accent, c.border, c.muted]

  return art.map((text, i): BannerRow => [[p[gradient[i]!] ?? c.muted, text]])
}

export const LOGO_WIDTH = Math.max(...LOGO_ART.map(line => line.length))
export const CADUCEUS_WIDTH = Math.max(...CADUCEUS_ART.map(line => line.length))

export const logo = (c: ThemeColors, customLogo?: string): BannerRow[] =>
  customLogo ? parseRichMarkup(customLogo) : colorize(LOGO_ART, LOGO_GRADIENT, c)

export const caduceus = (c: ThemeColors, customHero?: string): BannerRow[] =>
  customHero ? parseRichMarkup(customHero) : colorize(CADUCEUS_ART, CADUC_GRADIENT, c)

export const artWidth = (lines: BannerRow[]) =>
  lines.reduce(
    (m, row) =>
      Math.max(
        m,
        row.reduce((w, [, t]) => w + t.length, 0)
      ),
    0
  )
