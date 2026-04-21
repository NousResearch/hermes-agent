import type { ScrollBoxHandle } from '@hermes/ink'
import {
  type RefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore
} from 'react'

const ESTIMATE = 4
const OVERSCAN = 40
const MAX_MOUNTED = 260
const COLD_START = 40
const QUANTUM = OVERSCAN >> 1
const FREEZE_RENDERS = 2

const upperBound = (arr: number[], target: number) => {
  let lo = 0
  let hi = arr.length

  while (lo < hi) {
    const mid = (lo + hi) >> 1

    arr[mid]! <= target ? (lo = mid + 1) : (hi = mid)
  }

  return lo
}

export function useVirtualHistory(
  scrollRef: RefObject<ScrollBoxHandle | null>,
  items: readonly { key: string }[],
  columns: number,
  { estimate = ESTIMATE, overscan = OVERSCAN, maxMounted = MAX_MOUNTED, coldStartCount = COLD_START } = {}
) {
  const nodes = useRef(new Map<string, unknown>())
  const heights = useRef(new Map<string, number>())
  const refs = useRef(new Map<string, (el: unknown) => void>())
  const [ver, setVer] = useState(0)
  const [hasScrollRef, setHasScrollRef] = useState(false)
  const metrics = useRef({ sticky: true, top: 0, vp: 0 })

  // Width change: scale cached heights (not clear — clearing forces a
  // pessimistic back-walk mounting ~190 rows at once, each a fresh
  // marked.lexer + syntax highlight ≈ 3ms). Freeze mount range for 2
  // renders so warm memos survive; skip one measurement so useLayoutEffect
  // doesn't poison the scaled cache with pre-resize Yoga heights.
  const prevColumns = useRef(columns)
  const skipMeasurement = useRef(false)
  const prevRange = useRef<null | readonly [number, number]>(null)
  const freezeRenders = useRef(0)

  if (prevColumns.current !== columns && prevColumns.current > 0 && columns > 0) {
    const ratio = prevColumns.current / columns

    prevColumns.current = columns

    for (const [k, h] of heights.current) {
      heights.current.set(k, Math.max(1, Math.round(h * ratio)))
    }

    skipMeasurement.current = true
    freezeRenders.current = FREEZE_RENDERS
  }

  useLayoutEffect(() => {
    setHasScrollRef(Boolean(scrollRef.current))
  }, [scrollRef])

  useSyncExternalStore(
    useCallback(
      (cb: () => void) => (hasScrollRef ? scrollRef.current?.subscribe(cb) : null) ?? (() => () => {}),
      [hasScrollRef, scrollRef]
    ),
    () => {
      const s = scrollRef.current

      if (!s) {
        return NaN
      }

      const b = Math.floor(s.getScrollTop() / QUANTUM)

      return s.isSticky() ? -b - 1 : b
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
      setVer(v => v + 1)
    }
  }, [items])

  const offsets = useMemo(() => {
    void ver
    const out = new Array<number>(items.length + 1).fill(0)

    for (let i = 0; i < items.length; i++) {
      out[i + 1] = out[i]! + Math.max(1, Math.floor(heights.current.get(items[i]!.key) ?? estimate))
    }

    return out
  }, [estimate, items, ver])

  const n = items.length
  const total = offsets[n] ?? 0
  const top = Math.max(0, scrollRef.current?.getScrollTop() ?? 0)
  const vp = Math.max(0, scrollRef.current?.getViewportHeight() ?? 0)
  const sticky = scrollRef.current?.isSticky() ?? true

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
    } else {
      start = Math.max(0, Math.min(n - 1, upperBound(offsets, Math.max(0, top - overscan)) - 1))
      end = Math.max(start + 1, Math.min(n, upperBound(offsets, top + vp + overscan)))
    }
  }

  if (end - start > maxMounted) {
    sticky ? (start = Math.max(0, end - maxMounted)) : (end = Math.min(n, start + maxMounted))
  }

  if (freezeRenders.current > 0) {
    freezeRenders.current--
  } else {
    prevRange.current = [start, end]
  }

  const measureRef = useCallback((key: string) => {
    let fn = refs.current.get(key)

    if (!fn) {
      fn = (el: unknown) => (el ? nodes.current.set(key, el) : nodes.current.delete(key))
      refs.current.set(key, fn)
    }

    return fn
  }, [])

  useLayoutEffect(() => {
    let dirty = false

    if (skipMeasurement.current) {
      skipMeasurement.current = false
    } else {
      for (let i = start; i < end; i++) {
        const k = items[i]?.key

        if (!k) {
          continue
        }

        const h = Math.ceil((nodes.current.get(k) as MeasuredNode | undefined)?.yogaNode?.getComputedHeight?.() ?? 0)

        if (h > 0 && heights.current.get(k) !== h) {
          heights.current.set(k, h)
          dirty = true
        }
      }
    }

    const s = scrollRef.current

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
      setVer(v => v + 1)
    }
  }, [end, hasScrollRef, items, scrollRef, start])

  return {
    bottomSpacer: Math.max(0, total - (offsets[end] ?? total)),
    end,
    measureRef,
    offsets,
    start,
    topSpacer: offsets[start] ?? 0
  }
}

interface MeasuredNode {
  yogaNode?: { getComputedHeight?: () => number } | null
}
