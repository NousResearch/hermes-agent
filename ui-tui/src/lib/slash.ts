import type { SlashCatalog } from '../types.js'

/** Match SlashCommandCompleter: command names, subcommands, then skills. */
export function paletteForLine(line: string, c: SlashCatalog | null): [string, string][] {
  if (!c || !line.startsWith('/')) {
    return []
  }

  const parts = line.split(/\s+/)
  const baseRaw = parts[0]!
  const base = baseRaw.toLowerCase()
  const inSub = parts.length > 1 || (parts.length === 1 && line.endsWith(' '))

  if (inSub) {
    const subText = parts.length > 1 ? parts.slice(1).join(' ') : ''

    if (subText.includes('  ') || parts.length > 2) {
      return []
    }

    const head = subText.split(/\s+/)[0] ?? ''

    if (subText.includes(' ') && head !== subText) {
      return []
    }

    const canonical = c.canon[base] ?? baseRaw
    const subs = c.sub[canonical]

    if (!subs?.length) {
      return []
    }

    const lo = head.toLowerCase()

    return subs
      .filter(s => s.toLowerCase().startsWith(lo) && s.toLowerCase() !== lo)
      .slice(0, 14)
      .map(s => [s, ''])
  }

  const word = line.slice(1)

  return c.pairs
    .filter(([k]) => k.slice(1).startsWith(word))
    .slice(0, 16)
    .map(([k, d]) => [k, d])
}

/** Tab: longest common prefix of palette matches, or first unique completion + space. */
export function tabAdvance(line: string, c: SlashCatalog | null): string | null {
  if (!c || !line.startsWith('/')) {
    return null
  }

  const rows = paletteForLine(line, c)

  if (!rows.length) {
    return null
  }

  const parts = line.split(/\s+/)
  const baseRaw = parts[0]!
  const base = baseRaw.toLowerCase()
  const inSub = parts.length > 1 || (parts.length === 1 && line.endsWith(' '))

  if (inSub) {
    const subText = parts.length > 1 ? parts.slice(1).join(' ') : ''
    const head = subText.split(/\s+/)[0] ?? ''
    const picks = rows.map(([s]) => s)

    if (picks.length === 1) {
      return `${baseRaw} ${picks[0]!} `
    }

    const cp = commonPrefix(picks)

    if (cp.length > head.length) {
      return `${baseRaw} ${cp}`
    }

    return null
  }

  const word = line.slice(1)
  const names = rows.map(([k]) => k.slice(1))
  const cp = commonPrefix(names)

  if (names.length === 1) {
    return `/${names[0]!} `
  }

  if (cp.length > word.length) {
    return `/${cp}`
  }

  return null
}

function commonPrefix(xs: string[]): string {
  if (!xs.length) {
    return ''
  }

  let n = 0

  outer: while (true) {
    const ch = xs[0]![n]

    if (ch === undefined) {
      break
    }

    for (const x of xs) {
      if (x[n] !== ch) {
        break outer
      }
    }

    n++
  }

  return xs[0]!.slice(0, n)
}
