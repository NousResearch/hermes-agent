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

const upperBound = (arr: number[], target: number) => {
  let lo = 0,
    hi = arr.length

  while (lo < hi) {
    const mid = (lo + hi) >> 1
    arr[mid]! <= target ? (lo = mid + 1) : (hi = mid)
  }

  return lo
}

export function useVirtualHistory(
  scrollRef: RefObject<ScrollBoxHandle | null>,
  items: readonly { key: string }[],
  { estimate = ESTIMATE, overscan = OVERSCAN, maxMounted = MAX_MOUNTED, coldStartCount = COLD_START } = {}
) {
  const nodes = useRef(new Map<string, any>())
  const heights = useRef(new Map<string, number>())
  const refs = useRef(new Map<string, (el: any) => void>())
  const [ver, setVer] = useState(0)

  useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => () => {}), [scrollRef]),
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
    const out = new Array<number>(items.length + 1).fill(0)

    for (let i = 0; i < items.length; i++) {
      out[i + 1] = out[i]! + Math.max(1, Math.floor(heights.current.get(items[i]!.key) ?? estimate))
    }

    return out
  }, [estimate, items, ver])

  const total = offsets[items.length] ?? 0
  const top = Math.max(0, scrollRef.current?.getScrollTop() ?? 0)
  const vp = Math.max(0, scrollRef.current?.getViewportHeight() ?? 0)
  const sticky = scrollRef.current?.isSticky() ?? true

  let start = 0,
    end = items.length

  if (items.length > 0) {
    if (vp <= 0) {
      start = Math.max(0, items.length - coldStartCount)
    } else {
      start = Math.max(0, Math.min(items.length - 1, upperBound(offsets, Math.max(0, top - overscan)) - 1))
      end = Math.max(start + 1, Math.min(items.length, upperBound(offsets, top + vp + overscan)))
    }
  }

  if (end - start > maxMounted) {
    sticky ? (start = Math.max(0, end - maxMounted)) : (end = Math.min(items.length, start + maxMounted))
  }

  const measureRef = useCallback((key: string) => {
    let fn = refs.current.get(key)

    if (!fn) {
      fn = (el: any) => (el ? nodes.current.set(key, el) : nodes.current.delete(key))
      refs.current.set(key, fn)
    }

    return fn
  }, [])

  useLayoutEffect(() => {
    let dirty = false

    for (let i = start; i < end; i++) {
      const k = items[i]?.key

      if (!k) {
        continue
      }

      const h = Math.ceil(nodes.current.get(k)?.yogaNode?.getComputedHeight?.() ?? 0)

      if (h > 0 && heights.current.get(k) !== h) {
        heights.current.set(k, h)
        dirty = true
      }
    }

    if (dirty) {
      setVer(v => v + 1)
    }
  }, [end, items, start])

  return {
    start,
    end,
    offsets,
    topSpacer: offsets[start] ?? 0,
    bottomSpacer: Math.max(0, total - (offsets[end] ?? total)),
    measureRef
  }
}
