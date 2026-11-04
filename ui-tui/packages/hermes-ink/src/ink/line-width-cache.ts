import { lruEvict } from './lru.js'
import { stringWidth } from './stringWidth.js'

// During streaming, text grows but completed lines are immutable.
// Caching stringWidth per-line avoids re-measuring hundreds of
// unchanged lines on every token (~50x reduction in stringWidth calls).
const cache = new Map<string, number>()

const MAX_CACHE_SIZE = 4096

export function lineWidth(line: string): number {
  const cached = cache.get(line)

  if (cached !== undefined) {
    cache.delete(line)
    cache.set(line, cached)

    return cached
  }

  const width = stringWidth(line)

  if (cache.size >= MAX_CACHE_SIZE) {
    cache.delete(cache.keys().next().value!)
  }

  cache.set(line, width)

  return width
}

export function lineWidthCacheSize(): number {
  return cache.size
}

export function evictLineWidthCache(keepRatio = 0): void {
  lruEvict(cache, keepRatio)
}
