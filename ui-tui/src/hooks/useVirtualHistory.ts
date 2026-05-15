import type { ScrollBoxHandle } from '@hermes/ink'
import {
  type RefObject,
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore
} from 'react'

const ESTIMATE = 4
// Overscan was 40 (= viewport) which is way more than needed when heights
// are well-estimated.  Cutting in half saves ~20 mounted items per scroll
// edge → smaller fiber tree → less buffer-compose work per frame.  HN/CC
// dev (https://news.ycombinator.com/item?id=46699072) confirmed GC pressure
// from large JSX trees was their main perf issue post-rewrite.
const OVERSCAN = 20
// Hard cap on mounted items.  Was 260; profiling showed ~23k live Yoga
// nodes during sustained PageUp catch-up (renderer p99=106ms).  The
// viewport+2*overscan = 80 rows of needed coverage = ~25 items at avg 3
// rows/item, so 120 leaves >4× headroom and never blanks the viewport
// even when items are tiny.
const MAX_MOUNTED = 120
const COLD_START = 30
// Floor on unmeasured row height used when computing coverage — guarantees
// the mounted span physically reaches the viewport bottom regardless of how
// small items actually are (at the cost of over-mounting when items are
// larger; overscan absorbs that).
const PESSIMISTIC = 1
// Tightest safe scrollTop bin for the useSyncExternalStore snapshot. Small
// wheel ticks that don't cross a bin short-circuit React's commit entirely;
// Ink keeps painting via ScrollBox.forceRender + direct scrollTop reads.
// Half of OVERSCAN keeps ≥20 rows of cushion before the mounted range
// would actually need to shift.
const QUANTUM = OVERSCAN >> 1
// Renders to keep the mount range frozen after width change (heights scaled
// but not yet re-measured). Render #1 skips measurement so pre-resize Yoga
// doesn't poison the scaled cache; render #2's useLayoutEffect captures
// post-resize heights; render #3 recomputes range with accurate data.
const FREEZE_RENDERS = 2
// Cap on NEW items mounted per commit when scrolling fast. Without this,
// a single PageUp into unmeasured territory mounts ~190 rows with
// PESSIMISTIC=1 coverage — each row running marked lexer + syntax
// highlighting for ~3ms = ~600ms sync block. Sliding toward the target
// over several commits keeps per-commit mount cost bounded.  Tightened
// from 25 → 12: each new item adds ~100 fibers / Yoga nodes, and a
// 25-item commit was the dominant contributor to the 100ms+ p99 frames.
const SLIDE_STEP = 12

const NOOP = () => {}

const upperBound = (arr: ArrayLike<number>, target: number, length = arr.length) => {
  let lo = 0
  let hi = length

  while (lo < hi) {
    const mid = (lo + hi) >> 1

    arr[mid]! <= target ? (lo = mid + 1) : (hi = mid)
  }

  return lo
}

export const shouldSetVirtualClamp = ({
  itemCount,
  liveTailActive = false,
  sticky,
  viewportHeight
}: {
  itemCount: number
  liveTailActive?: boolean
  sticky: boolean
  viewportHeight: number
}) => itemCount > 0 && viewportHeight > 0 && !sticky && !liveTailActive

export const ensureVirtualItemHeight = (
  heights: Map<string, number>,
  key: string,
  index: number,
  estimate: number,
  estimateHeight?: (index: number, key: string) => number
) => {
  const cached = heights.get(key)

  if (cached !== undefined) {
    return Math.max(1, Math.floor(cached))
  }

  const seeded = Math.max(1, Math.floor(estimateHeight?.(index, key) ?? estimate))
  heights.set(key, seeded)

  return seeded
}

export function useVirtualHistory(
  scrollRef: RefObject<ScrollBoxHandle | null>,
  items: readonly { key: string }[],
  columns: number,
  {
    estimate = ESTIMATE,
    estimateHeight,
    initialHeights,
    liveTailActive = false,
    onHeightsChange,
    overscan = OVERSCAN,
    maxMounted = MAX_MOUNTED,
    coldStartCount = COLD_START
  }: VirtualHistoryOptions = {}
) {
  const nodes = useRef(new Map<string, unknown>())
  const heights = useRef(new Map(initialHeights))
  const initialHeightsRef = useRef(initialHeights)
  const refs = useRef(new Map<string, (el: unknown) => void>())
  const onHeightsChangeRef = useRef(onHeightsChange)
  // Bump whenever heightCache mutates so offsets rebuild on next read.
  // Ref (not state) — checked during render phase, zero extra commits.
  const offsetVersion = useRef(0)

  // Cached offsets: reused Float64Array keyed on (itemCount, version) so we
  // only rebuild when something actually changed. Previous approach allocated
  // a fresh Array(n+1) every render — at n=10k that's ~80KB/render of GC
  // pressure during streaming.
  const offsetsCache = useRef<{ arr: Float64Array; n: number; version: number }>({
    arr: new Float64Array(0),
    n: -1,
    version: -1
  })

  const [hasScrollRef, setHasScrollRef] = useState(false)
  // Height cache writes happen in layout effects; bump once so offsets and
  // clamp bounds rebuild without waiting for the next scroll/input event.
  const [measuredHeightVersion, bumpMeasuredHeightVersion] = useState(0)
  const metrics = useRef({ sticky: true, top: 0, vp: 0 })
  const lastScrollTopRef = useRef(0)

  // Width change: scale cached heights by oldCols/newCols instead of clearing
  // (clearing forces a pessimistic back-walk mounting ~190 rows at once, each
  // a fresh marked.lexer + syntax highlight ≈ 3ms). Freeze the mount range
  // for 2 renders so warm memos survive; skip one measurement pass so
  // useLayoutEffect doesn't poison the scaled cache with pre-resize Yoga
  // heights.
  const prevColumns = useRef(columns)
  const skipMeasurement = useRef(false)
  const prevRange = useRef<null | readonly [number, number]>(null)
  const freezeRenders = useRef(0)

  onHeightsChangeRef.current = onHeightsChange

  if (initialHeightsRef.current !== initialHeights) {
    initialHeightsRef.current = initialHeights
    heights.current = new Map(initialHeights)
    offsetVersion.current++
  }

  if (prevColumns.current !== columns && prevColumns.current > 0 && columns > 0) {
    const ratio = prevColumns.current / columns

    prevColumns.current = columns

    for (const [k, h] of heights.current) {
      heights.current.set(k, Math.max(1, Math.round(h * ratio)))
    }

    offsetVersion.current++
    skipMeasurement.current = true
    freezeRenders.current = FREEZE_RENDERS
  }

  useLayoutEffect(() => {
    setHasScrollRef(Boolean(scrollRef.current))
  }, [scrollRef])

  // Quantized snapshot: same-bin scrolls (most wheel ticks) produce the same
  // number → React.Object.is short-circuits the commit entirely. sticky state
  // is folded in via the sign bit so sticky→broken transitions also trigger.
  // Uses the TARGET (committed + pendingDelta), not committed scrollTop, so
  // scrollBy notifications immediately remount for the destination before
  // Ink's drain frames need the children.
  const subscribe = useCallback(
    (cb: () => void) => (hasScrollRef ? scrollRef.current?.subscribe(cb) : null) ?? NOOP,
    [hasScrollRef, scrollRef]
  )

  useSyncExternalStore(
    subscribe,
    () => {
      const s = scrollRef.current

      if (!s) {
        return NaN
      }

      const target = s.getScrollTop() + s.getPendingDelta()
      const bin = Math.floor(target / QUANTUM)

      return s.isSticky() ? ~bin : bin
    },
    () => NaN
  )

  useEffect(() => {
    const keep = new Set(items.map(i => i.key))
    let dirty = false

    for (const k of heights.current.keys()) {
      if (!keep.has(k)) {
        heights.current.delete(k)
        nodes.current.delete(k)
        refs.current.delete(k)
        dirty = true
      }
    }

    if (dirty) {
      offsetVersion.current++
    }
  }, [items])

  // Offsets: Float64Array reused across renders, invalidated by offsetVersion
  // bumps from heightCache writers (measureRef, resize-scale, GC). Binary
  // search tolerates either monotone source, so no need to rebuild unless
  // something changed.
  const n = items.length

  if (offsetsCache.current.version !== offsetVersion.current || offsetsCache.current.n !== n) {
    const arr = offsetsCache.current.arr.length >= n + 1 ? offsetsCache.current.arr : new Float64Array(n + 1)

    arr[0] = 0

    for (let i = 0; i < n; i++) {
      arr[i + 1] = arr[i]! + ensureVirtualItemHeight(heights.current, items[i]!.key, i, estimate, estimateHeight)
    }

    offsetsCache.current = { arr, n, version: offsetVersion.current }
  }

  const offsets = offsetsCache.current.arr
  const total = offsets[n] ?? 0
  const top = Math.max(0, scrollRef.current?.getScrollTop() ?? 0)
  const pendingDelta = scrollRef.current?.getPendingDelta() ?? 0
  const target = Math.max(0, top + pendingDelta)
  const vp = Math.max(0, scrollRef.current?.getViewportHeight() ?? 0)
  const sticky = scrollRef.current?.isSticky() ?? true
  const recentManual = Date.now() - (scrollRef.current?.getLastManualScrollAt() ?? 0) < 1200

  // During a freeze, drop the frozen range if items shrank past its start
  // (/clear, compaction) — clamping would collapse to an empty mount and
  // flash blank. Fall through to the normal path in that case.
  const frozenRange =
    freezeRenders.current > 0 && prevRange.current && prevRange.current[0] < n ? prevRange.current : null

  let start = 0
  let end = n

  if (frozenRange) {
    start = frozenRange[0]
    end = Math.min(frozenRange[1], n)
  } else if (n > 0) {
    if (vp <= 0) {
      start = Math.max(0, n - coldStartCount)
    } else if (sticky && !recentManual) {
      const budget = vp + overscan
      start = n

      while (start > 0 && total - offsets[start - 1]! < budget) {
        start--
      }
    } else {
      // User scrolled up. Span [committed..target] so every drain frame is
      // covered. Claude-code caps the span at 3×viewport so pendingDelta
      // growing unbounded (MX Master free-spin) doesn't blow the mount
      // budget; the clamp (setClampBounds) shows edge-of-mounted content
      // during catch-up.
      const MAX_SPAN = vp * 3
      const rawLo = Math.min(top, target)
      const rawHi = Math.max(top, target)
      const span = rawHi - rawLo
      const clampedLo = span > MAX_SPAN ? (pendingDelta < 0 ? rawHi - MAX_SPAN : rawLo) : rawLo
      const clampedHi = clampedLo + Math.min(span, MAX_SPAN)
      const lo = Math.max(0, clampedLo - overscan)
      const hi = clampedHi + vp + overscan

      // Binary search — offsets is monotone. Linear walk was O(n) at n=10k+,
      // ~2ms per render during scroll.
      start = Math.max(0, Math.min(n - 1, upperBound(offsets, lo, n + 1) - 1))
      end = Math.max(start + 1, Math.min(n, upperBound(offsets, hi, n + 1)))
    }
  }

  if (end - start > maxMounted) {
    sticky ? (start = Math.max(0, end - maxMounted)) : (end = Math.min(n, start + maxMounted))
  }

  // Coverage guarantee: ensure sum(real or pessimistic heights) ≥
  // viewportH + 2*overscan so the viewport is physically covered even when
  // items are tiny. Pessimistic because uncached items use a floor of 1 —
  // over-mounts when items are large, never leaves blank spacer showing.
  if (n > 0 && vp > 0 && !frozenRange) {
    const needed = vp + 2 * overscan
    let coverage = 0

    for (let i = start; i < end; i++) {
      coverage += ensureVirtualItemHeight(heights.current, items[i]!.key, i, PESSIMISTIC, estimateHeight)
    }

    if (sticky) {
      const minStart = Math.max(0, end - maxMounted)

      while (start > minStart && coverage < needed) {
        start--
        coverage += ensureVirtualItemHeight(heights.current, items[start]!.key, start, PESSIMISTIC, estimateHeight)
      }
    } else {
      const maxEnd = Math.min(n, start + maxMounted)

      while (end < maxEnd && coverage < needed) {
        coverage += ensureVirtualItemHeight(heights.current, items[end]!.key, end, PESSIMISTIC, estimateHeight)
        end++
      }
    }
  }

  // Slide cap: limit how many NEW items mount this commit. Gates on scroll
  // VELOCITY (|scrollTop delta since last commit| + |pendingDelta| >
  // 2×viewport — key-repeat PageUp moves ~viewport/2 per press). Covers
  // both scrollBy (pendingDelta) and scrollTo (direct write). Normal single
  // PageUp skips this; the clamp holds the viewport at the mounted edge
  // during catch-up so there's no blank screen. Only caps range GROWTH;
  // shrinking is unbounded.
  if (!frozenRange && prevRange.current && vp > 0) {
    const velocity = Math.abs(top - lastScrollTopRef.current) + Math.abs(pendingDelta)

    if (velocity > vp * 2) {
      const [pS, pE] = prevRange.current

      start = Math.max(start, pS - SLIDE_STEP)
      end = Math.min(end, pE + SLIDE_STEP)

      // A large jump past the capped end can invert (start > end); mount
      // SLIDE_STEP items from the new start so the viewport isn't blank
      // during catch-up.
      if (start > end) {
        end = Math.min(start + SLIDE_STEP, n)
      }
    }
  }

  lastScrollTopRef.current = top

  if (freezeRenders.current > 0) {
    freezeRenders.current--
  } else {
    prevRange.current = [start, end]
  }

  // Time-slice range growth via useDeferredValue. Urgent render keeps Ink
  // painting with the OLD range (all memo hits, fast); deferred render
  // transitions to the NEW range (fresh mounts: Md, syntax highlight) in a
  // non-blocking background commit. The clamp (setClampBounds) pins the
  // viewport to the mounted edge so there's no visual artifact from the
  // deferred range lagging briefly. Only deferral range GROWTH — shrinking
  // is cheap (unmount = remove fiber, no parse).
  const dStart = useDeferredValue(start)
  const dEnd = useDeferredValue(end)
  let effStart = start < dStart ? dStart : start
  let effEnd = end > dEnd ? dEnd : end

  // Inverted range (large jump with deferred value lagging) or sticky snap
  // (scrollToBottom needs the tail mounted NOW so maxScroll lands on content,
  // not bottomSpacer) — skip deferral.
  if (effStart > effEnd || sticky) {
    effStart = start
    effEnd = end
  }

  // Scrolling DOWN — bypass effEnd deferral so the tail mounts immediately.
  // Without this, the clamp holds scrollTop short of the real bottom and
  // the user feels "stuck before bottom". effStart stays deferred so scroll-
  // UP keeps time-slicing (older messages parse on mount).
  if (pendingDelta > 0) {
    effEnd = end
  }

  // Final O(viewport) enforcement. Deferred+bypass combinations above can
  // leak: during sustained PageUp, concurrent mode interleaves dStart updates
  // with effEnd=end bypasses across commits and the effective window drifts
  // wider than either bound alone. Trim the far edge by viewport position
  // (not pendingDelta direction — that flips mid-settle under concurrent
  // scheduling and yanks scrollTop).
  if (effEnd - effStart > maxMounted && vp > 0) {
    const mid = (offsets[effStart]! + offsets[effEnd]!) / 2

    if (top < mid) {
      effEnd = effStart + maxMounted
    } else {
      effStart = effEnd - maxMounted
    }
  }

  const measureRef = useCallback((key: string) => {
    let fn = refs.current.get(key)

    if (!fn) {
      fn = (el: unknown) => {
        if (el) {
          nodes.current.set(key, el)

          return
        }

        // Measure-at-unmount: the yogaNode is still valid here (reconciler
        // calls ref(null) before removeChild → freeRecursive), so we grab
        // the final height before WASM release. Without this, items
        // scrolled out during fast pan keep a stale estimate in heightCache
        // and offset math drifts until the next mount/remount cycle.
        const existing = nodes.current.get(key) as MeasuredNode | undefined
        const h = Math.ceil(existing?.yogaNode?.getComputedHeight?.() ?? 0)

        if (h > 0 && heights.current.get(key) !== h) {
          heights.current.set(key, h)
          offsetVersion.current++
          onHeightsChangeRef.current?.(heights.current)
        }

        nodes.current.delete(key)
      }

      refs.current.set(key, fn)
    }

    return fn
  }, [])

  useLayoutEffect(() => {
    const s = scrollRef.current
    let dirty = false
    let heightDirty = false

    // Give the renderer the mounted-row coverage for passive scroll clamping.
    // Clamp MUST use the EFFECTIVE (deferred) range, not the immediate one.
    // During fast scroll, immediate [start,end] may already cover the new
    // scrollTop position, but children still render at the deferred range.
    // If clamp used immediate bounds, render-node-to-output's drain-gate
    // would drain past the deferred children's span → viewport lands in
    // spacer → white flash.
    if (s && shouldSetVirtualClamp({ itemCount: n, liveTailActive, sticky, viewportHeight: vp })) {
      const effTopSpacer = offsets[effStart] ?? 0
      const effBottom = offsets[effEnd] ?? total
      // At effEnd=n there's no bottomSpacer — use Infinity so render-node-
      // to-output's own Math.min(cur, maxScroll) governs. Using offsets[n]
      // here would bake in heightCache (one render behind Yoga), and during
      // streaming the tail item's cached height lags its real height —
      // sticky-break would then clamp below the real max and push
      // streaming text off-viewport.
      const clampMin = effStart === 0 ? 0 : effTopSpacer
      const clampMax = effEnd === n ? Infinity : Math.max(effTopSpacer, effBottom - vp)

      s.setClampBounds(clampMin, clampMax)
    } else {
      s?.setClampBounds(undefined, undefined)
    }

    if (skipMeasurement.current) {
      skipMeasurement.current = false
    } else {
      for (let i = effStart; i < effEnd; i++) {
        const k = items[i]?.key

        if (!k) {
          continue
        }

        const h = Math.ceil((nodes.current.get(k) as MeasuredNode | undefined)?.yogaNode?.getComputedHeight?.() ?? 0)

        if (h > 0 && heights.current.get(k) !== h) {
          heights.current.set(k, h)
          dirty = true
          heightDirty = true
        }
      }
    }

    if (s) {
      const next = {
        sticky: s.isSticky(),
        top: Math.max(0, s.getScrollTop() + s.getPendingDelta()),
        vp: Math.max(0, s.getViewportHeight())
      }

      if (
        next.sticky !== metrics.current.sticky ||
        next.top !== metrics.current.top ||
        next.vp !== metrics.current.vp
      ) {
        metrics.current = next
        dirty = true
      }
    }

    if (dirty) {
      offsetVersion.current++
      onHeightsChangeRef.current?.(heights.current)
    }

    if (heightDirty) {
      bumpMeasuredHeightVersion(n => n + 1)
    }
  }, [effEnd, effStart, items, liveTailActive, measuredHeightVersion, n, offsets, scrollRef, sticky, total, vp])

  return {
    bottomSpacer: Math.max(0, total - (offsets[effEnd] ?? total)),
    end: effEnd,
    measureRef,
    offsets,
    start: effStart,
    topSpacer: offsets[effStart] ?? 0
  }
}

interface MeasuredNode {
  yogaNode?: { getComputedHeight?: () => number } | null
}

interface VirtualHistoryOptions {
  coldStartCount?: number
  estimate?: number
  estimateHeight?: (index: number, key: string) => number
  initialHeights?: ReadonlyMap<string, number>
  liveTailActive?: boolean
  maxMounted?: number
  onHeightsChange?: (heights: ReadonlyMap<string, number>) => void
  overscan?: number
}
