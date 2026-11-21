import { stripAnsi } from '@hermes/shared/ansi'
import { atom } from 'nanostores'

/** The transcript row `/find` has selected: scrolled into view and marked like a rewind focus.
 *  -1 while searching with no match; null when `/find` is closed. */
export const $findFocus = atom<null | number>(null)

/** Set by Esc just before `/find` closes: the transcript returns to where it was before searching. */
export const $findScrollRestore = atom(false)

interface FindRow {
  kind?: string
  text: string
}

const squash = (text: string) => stripAnsi(text).replace(/\s+/g, ' ').trim().toLowerCase()

// The banner, and the echo of a typed slash command (`/find …` itself would always be the newest hit).
const UNSEARCHED = new Set(['intro', 'slash'])

/** Indices of the rows whose text contains `query`, ignoring case and runs of whitespace (a match may
 *  span a line break). */
export function findTranscriptHits(rows: readonly FindRow[], query: string): number[] {
  const want = squash(query)

  if (!want) {
    return []
  }

  return rows.flatMap((row, index) =>
    !UNSEARCHED.has(row.kind ?? '') && squash(row.text).includes(want) ? [index] : []
  )
}

/** One line of `text` around the first match, so a hit is readable without scrolling (and in
 *  `--tui-native`, where the transcript cannot scroll). */
export function findSnippet(text: string, query: string, width: number): string {
  const flat = stripAnsi(text).replace(/\s+/g, ' ').trim()
  const at = flat.toLowerCase().indexOf(squash(query))
  const start = flat.length <= width ? 0 : Math.max(0, Math.min(at - Math.floor(width / 3), flat.length - width))
  const snippet = flat.slice(start, start + width)

  return `${start > 0 ? '…' : ''}${snippet}${start + width < flat.length ? '…' : ''}`
}
