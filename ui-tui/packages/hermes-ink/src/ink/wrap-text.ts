import sliceAnsi from '../utils/sliceAnsi.js'

import { lruEvict } from './lru.js'
import { stringWidth } from './stringWidth.js'
import type { Styles } from './styles.js'
import { wrapAnsi } from './wrapAnsi.js'

const ELLIPSIS = '…'

// CPU profile (Apr 2026) showed `wrap-ansi` → `string-width` consuming 30% of
// total runtime during fast scroll: every layout pass re-wraps every visible
// line via wrap-ansi, which calls string-width once per grapheme. The output
// is pure of (text, maxWidth, wrapType), so memoize it. LRU-bounded so long
// sessions don't accrete unbounded cache.
const WRAP_CACHE_LIMIT = 4096

/** Memoized wrap result. `trimmed[i]` is true when the newline before output
 *  line i+1 is a soft-wrap boundary where wrap-trim dropped the single
 *  separator space (hard mid-word splits and hard `\n` are false). The
 *  selection copier re-inserts exactly those spaces so drag-copy round-trips
 *  the source text. Aligned with `text.split('\n')`: line i+1 reads
 *  `trimmed[i]` (empty for modes that add no newlines). */
export type WrapEntry = { text: string; trimmed: boolean[] }

const wrapCache = new Map<string, WrapEntry>()

function memoizedEntry(text: string, maxWidth: number, wrapType: Styles['textWrap']): WrapEntry {
  // Key folds maxWidth + wrapType into the prefix so the same text re-wrapped
  // at a different width doesn't collide. Width prefix bounded by viewport
  // (~10 distinct widths in a session); wrapType bounded by enum (~6 values).
  const key = `${maxWidth}|${wrapType}|${text}`
  const cached = wrapCache.get(key)

  if (cached !== undefined) {
    // LRU touch
    wrapCache.delete(key)
    wrapCache.set(key, cached)

    return cached
  }

  const result = computeEntry(text, maxWidth, wrapType)

  if (wrapCache.size >= WRAP_CACHE_LIMIT) {
    wrapCache.delete(wrapCache.keys().next().value!)
  }

  wrapCache.set(key, result)

  return result
}

// sliceAnsi may include a boundary-spanning wide char (e.g. CJK at position
// end-1 with width 2 overshoots by 1). Retry with a tighter bound once.
function sliceFit(text: string, start: number, end: number): string {
  const s = sliceAnsi(text, start, end)

  return stringWidth(s) > end - start ? sliceAnsi(text, start, end - 1) : s
}

function truncate(text: string, columns: number, position: 'start' | 'middle' | 'end'): string {
  if (columns < 1) {
    return ''
  }

  if (columns === 1) {
    return ELLIPSIS
  }

  const length = stringWidth(text)

  if (length <= columns) {
    return text
  }

  if (position === 'start') {
    return ELLIPSIS + sliceFit(text, length - columns + 1, length)
  }

  if (position === 'middle') {
    const half = Math.floor(columns / 2)

    return sliceFit(text, 0, half) + ELLIPSIS + sliceFit(text, length - (columns - half) + 1, length)
  }

  return sliceFit(text, 0, columns - 1) + ELLIPSIS
}

/** Wrap one source line, recording per-boundary trim flags. `trimmed[i]` is
 *  true exactly when the old loop dropped the single separator space between
 *  piece i and i+1 (a hard mid-word split records false, so the copier keeps
 *  it glued). The mutation order matches the old loop exactly. */
function splitTrimWrap(line: string, maxWidth: number): WrapEntry {
  const pieces = wrapAnsi(line, maxWidth, { trim: false, hard: true }).split('\n')
  const trimmed: boolean[] = []

  for (let index = 0; index < pieces.length - 1; index++) {
    const current = pieces[index]!
    const next = pieces[index + 1]!

    if (/\s$/.test(current)) {
      pieces[index] = current.replace(/\s$/, '')
      trimmed.push(true)
    } else if (/^\s/.test(next)) {
      pieces[index + 1] = next.replace(/^\s/, '')
      trimmed.push(true)
    } else {
      trimmed.push(false)
    }
  }

  return { text: pieces.join('\n'), trimmed }
}

function computeEntry(text: string, maxWidth: number, wrapType: Styles['textWrap']): WrapEntry {
  if (wrapType === 'wrap') {
    const pieces = wrapAnsi(text, maxWidth, { trim: false, hard: true }).split('\n')

    return { text: pieces.join('\n'), trimmed: Array<boolean>(pieces.length - 1).fill(false) }
  }

  if (wrapType === 'wrap-char') {
    const pieces = wrapAnsi(text, maxWidth, { trim: false, hard: true, wordWrap: false }).split('\n')

    return { text: pieces.join('\n'), trimmed: Array<boolean>(pieces.length - 1).fill(false) }
  }

  if (wrapType === 'wrap-trim') {
    const out: string[] = []
    const trimmed: boolean[] = []

    for (const line of text.split('\n')) {
      const entry = splitTrimWrap(line, maxWidth)

      if (out.length > 0) {
        // Hard source newline, never a trimmed soft boundary.
        trimmed.push(false)
      }

      out.push(entry.text)
      trimmed.push(...entry.trimmed)
    }

    return { text: out.join('\n'), trimmed }
  }

  if (wrapType!.startsWith('truncate')) {
    const position: 'end' | 'middle' | 'start' =
      wrapType === 'truncate-middle' ? 'middle' : wrapType === 'truncate-start' ? 'start' : 'end'

    return { text: truncate(text, maxWidth, position), trimmed: [] }
  }

  return { text, trimmed: [] }
}

export default function wrapText(text: string, maxWidth: number, wrapType: Styles['textWrap']): string {
  // Skip cache for trivial inputs (faster than Map lookup).
  if (!text || maxWidth <= 0) {
    return computeEntry(text, maxWidth, wrapType).text
  }

  return memoizedEntry(text, maxWidth, wrapType).text
}

/** Memoized wrap that also reports per-boundary trim flags (see WrapEntry).
 *  Shares one cache lookup with wrapText: a miss wraps once, not twice.
 *  // ponytail: flags ride the existing wrap cache; split the caches if
 *  profiling shows flag-only callers polluting string-hit rates. */
export function wrapTextWithTrim(text: string, maxWidth: number, wrapType: Styles['textWrap']): WrapEntry {
  if (!text || maxWidth <= 0) {
    return computeEntry(text, maxWidth, wrapType)
  }

  return memoizedEntry(text, maxWidth, wrapType)
}

export function wrapCacheSize(): number {
  return wrapCache.size
}

export function evictWrapCache(keepRatio = 0): void {
  lruEvict(wrapCache, keepRatio)
}
