import { type AnsiCode, ansiCodesToString, reduceAnsiCodes, tokenize, undoAnsiCodes } from '@alcalzone/ansi-tokenize'

import { lruEvict } from '../ink/lru.js'
import { stringWidth } from '../ink/stringWidth.js'

function isEndCode(code: AnsiCode): boolean {
  return code.code === code.endCode
}

function filterStartCodes(codes: AnsiCode[]): AnsiCode[] {
  return codes.filter(c => !isEndCode(c))
}

// LRU cache: same (string, start, end) → same output. Output.get() re-emits
// identical writes every frame for stable transcript content; this avoids
// re-tokenizing them. CPU profile (Apr 2026) showed sliceAnsi at 18% total
// time during scroll. Bounded at 4096 entries — entries are short clipped
// lines so memory cost is small.
const sliceCache = new Map<string, string>()
const SLICE_CACHE_LIMIT = 4096

export default function sliceAnsi(str: string, start: number, end?: number): string {
  if (!str) {
    return ''
  }

  // Hot-path: only cache when end is defined (the Output.get() use-case).
  if (end !== undefined) {
    const key = `${start}|${end}|${str}`
    const cached = sliceCache.get(key)

    if (cached !== undefined) {
      sliceCache.delete(key)
      sliceCache.set(key, cached)

      return cached
    }

    const result = computeSlice(str, start, end)

    if (sliceCache.size >= SLICE_CACHE_LIMIT) {
      sliceCache.delete(sliceCache.keys().next().value!)
    }

    sliceCache.set(key, result)

    return result
  }

  return computeSlice(str, start, end)
}

export function sliceCacheSize(): number {
  return sliceCache.size
}

export function evictSliceCache(keepRatio = 0): void {
  lruEvict(sliceCache, keepRatio)
}

function computeSlice(str: string, start: number, end?: number): string {
  const tokens = tokenize(str)
  let activeCodes: AnsiCode[] = []
  let position = 0
  let result = ''
  let include = false

  for (const token of tokens) {
    const width = token.type === 'ansi' ? 0 : token.fullWidth ? 2 : stringWidth(token.value)

    if (end !== undefined && position >= end) {
      if (token.type === 'ansi' || width > 0 || !include) {
        break
      }
    }

    if (token.type === 'ansi') {
      activeCodes.push(token)

      if (include) {
        result += token.code
      }
    } else {
      if (!include && position >= start) {
        if (start > 0 && width === 0) {
          continue
        }

        include = true
        activeCodes = filterStartCodes(reduceAnsiCodes(activeCodes))
        result = ansiCodesToString(activeCodes)
      }

      if (include) {
        result += token.value
      }

      position += width
    }
  }

  const activeStartCodes = filterStartCodes(reduceAnsiCodes(activeCodes))
  result += ansiCodesToString(undoAnsiCodes(activeStartCodes))

  return result
}
