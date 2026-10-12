// Unified cache eviction for the four hot Ink module-level caches:
//   - widthCache       (stringWidth.ts)
//   - wrapCache        (wrap-text.ts)
//   - sliceCache       (sliceAnsi.ts)
//   - lineWidthCache   (line-width-cache.ts)
//
// Used by the host (TUI) under memory pressure or on session swap to drop
// content-keyed entries that won't recur. All caches are content-keyed
// (not session-keyed), so cross-session sharing is normally beneficial —
// only evict when memory tightens or when the user explicitly resets.

import { evictSliceCache, sliceCacheSize } from '../utils/sliceAnsi.js'

import { evictLineWidthCache, lineWidthCacheSize } from './line-width-cache.js'
import { evictWidthCache, widthCacheSize } from './stringWidth.js'
import { evictWrapCache, wrapCacheSize } from './wrap-text.js'

export interface InkCacheSizes {
  lineWidth: number
  slice: number
  width: number
  wrap: number
}

function inkCacheSizes(): InkCacheSizes {
  return {
    lineWidth: lineWidthCacheSize(),
    slice: sliceCacheSize(),
    width: widthCacheSize(),
    wrap: wrapCacheSize()
  }
}

export type EvictLevel = 'all' | 'half'

export function evictInkCaches(level: EvictLevel = 'half'): InkCacheSizes {
  const keep = level === 'half' ? 0.5 : 0

  evictWidthCache(keep)
  evictWrapCache(keep)
  evictSliceCache(keep)
  evictLineWidthCache(keep)

  return inkCacheSizes()
}
