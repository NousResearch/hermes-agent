import { getHeapStatistics } from 'node:v8'

import { type HeapDumpResult, performHeapDump } from './memory.js'

export type MemoryLevel = 'critical' | 'high' | 'normal'

export interface MemorySnapshot {
  heapUsed: number
  level: MemoryLevel
  rss: number
}

export interface MemoryMonitorOptions {
  criticalBytes?: number
  highBytes?: number
  intervalMs?: number
  onCritical?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
  onHigh?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
  // Fired ONCE when heap growth looks abnormal while still far below the
  // critical exit threshold — the regime where the TUI used to die silently
  // (#34095: Node OOMs from an Ink render-tree blowup at a few hundred MB,
  // well under criticalBytes, so onCritical never fired and the gateway death
  // showed up only as a bare `stdin EOF`). A visible warning here makes that
  // class of death diagnosable instead of silent.
  onWarn?: (snap: MemorySnapshot) => void
  warnBytes?: number
  // STAGE 0 (proactive renderer GC): the warn regime (≈600MB, well below the
  // `high` watermark) is exactly where a long render-bound session lives —
  // heap climbs, GC churns, the Node parent pegs CPU, and input lags, but the
  // existing eviction (gated on `high` ≈70% of an 8GB ceiling) NEVER fires. So
  // a 1GB session warns forever and never gets relief. These hooks make the
  // warn regime ACT instead of only logging.
  //
  // onWarnRelief: invoked when warn pressure is detected, BEFORE onWarn, to
  // prune Ink content caches + force a GC pass. Returning the post-prune heap
  // (or void) lets the monitor decide if relief worked.
  onWarnRelief?: () => void | Promise<void>
  // onSustainedPressure: invoked when heap stays at/above warnBytes for
  // `sustainedTicks` consecutive ticks DESPITE relief — i.e. pruning didn't
  // help, this is a genuine render-tree blowup. The entry uses this to trigger
  // a graceful, seamless renderer recycle (Stage 1) rather than waiting for the
  // session to wedge or OOM. Fired at most once per sustained episode.
  onSustainedPressure?: (snap: MemorySnapshot) => void
  // Consecutive warn-regime ticks at/above warnBytes after relief before
  // onSustainedPressure fires. Default 6 (≈60s at the 10s interval).
  sustainedTicks?: number
}

const GB = 1024 ** 3
const MB = 1024 ** 2

// STAGE 0: force a V8 GC pass when the runtime exposes it (node --expose-gc).
// In the warn regime the heap is churning faster than incremental GC reclaims;
// a forced major GC after pruning Ink caches materially drops RSS and cuts the
// GC-thrash CPU. No-ops silently when global.gc is unavailable so this is safe
// in any runtime. Cheap: a single synchronous major GC every ≥10s tick.
function forceGcIfAvailable(): void {
  try {
    const gc = (globalThis as { gc?: () => void }).gc
    if (typeof gc === 'function') {
      gc()
    }
  } catch {
    // best-effort; never let a GC hiccup break the monitor tick
  }
}

// Resolve the exit / dump thresholds RELATIVE to the actual V8 heap ceiling
// (--max-old-space-size, 8GB for the TUI) instead of hardcoding 2.5GB. The old
// constant killed the process — and silently closed the gateway's stdin — at
// ~31% of an 8GB ceiling, treating a normal long-session heap as an OOM. We now
// exit only when genuinely near the ceiling (critical ~88%, high ~70%), and
// clamp to sane floors/ceilings so a tiny --max-old-space-size can't drive the
// thresholds below the warn watermark. Callers may still override explicitly.
function resolveThresholds(criticalBytes?: number, highBytes?: number) {
  let limit = 0
  try {
    limit = getHeapStatistics().heap_size_limit || 0
  } catch {
    limit = 0
  }

  // Fall back to the historical 8GB ceiling if V8 doesn't report one.
  const ceiling = limit > 0 ? limit : 8 * GB
  const critical = criticalBytes ?? Math.max(2 * GB, Math.round(ceiling * 0.88))
  const high = highBytes ?? Math.max(1 * GB, Math.min(critical - 256 * MB, Math.round(ceiling * 0.7)))

  return { critical, high }
}

// Deferred @hermes/ink import: loading `@hermes/ink` at module top-level
// pulls the full ~414KB Ink bundle (React, renderer, components, hooks) onto
// the critical path before the Python gateway can even be spawned. That
// serialised roughly 150ms of Node work in front of gw.start() on every
// cold `hermes --tui` launch.
//
// evictInkCaches only runs inside `tick()`, which fires on a 10s timer and
// only when heap pressure crosses the high-water mark — by then Ink has
// long since been loaded by the app entry. This dynamic import is a no-op
// on the hot path (module is already in the ESM cache); when a startup
// spike somehow trips the threshold before the app registers its own Ink
// import, we pay the load cost exactly once, inside the tick that needs it.
let _evictInkCaches: ((level: 'all' | 'half') => unknown) | null = null
let _evictInkCachesPromise: Promise<(level: 'all' | 'half') => unknown> | null = null

async function _ensureEvictInkCaches(): Promise<(level: 'all' | 'half') => unknown> {
  if (_evictInkCaches) {
    return _evictInkCaches
  }

  _evictInkCachesPromise ??= import('@hermes/ink')
    .then(mod => {
      _evictInkCaches = mod.evictInkCaches as (level: 'all' | 'half') => unknown

      return _evictInkCaches
    })
    .catch(err => {
      _evictInkCachesPromise = null
      throw err
    })

  return _evictInkCachesPromise
}

export function startMemoryMonitor({
  criticalBytes,
  highBytes,
  intervalMs = 10_000,
  onCritical,
  onHigh,
  onWarn,
  warnBytes = 600 * MB,
  onWarnRelief,
  onSustainedPressure,
  sustainedTicks = 6
}: MemoryMonitorOptions = {}): () => void {
  const { critical, high } = resolveThresholds(criticalBytes, highBytes)
  const dumped = new Set<Exclude<MemoryLevel, 'normal'>>()
  const inFlight = new Set<Exclude<MemoryLevel, 'normal'>>()

  // STAGE 0 proactive-GC state. `warnPressureTicks` counts consecutive ticks
  // at/above warnBytes (below `high`); once relief has run and pressure
  // persists for `sustainedTicks`, onSustainedPressure fires once. `reliefInFlight`
  // guards against overlapping async relief passes. `sustainedFired` re-arms
  // only after heap falls back below warnBytes.
  let warnPressureTicks = 0
  let reliefInFlight = false
  let sustainedFired = false

  // Early-warning state (#34095): the silent-death regime is BELOW `high`, so
  // the level machine above never sees it. Track the previous sample and fire
  // onWarn at most once when heap both crosses a modest absolute floor AND is
  // climbing steeply (≥150MB between 10s ticks) — the signature of a render-
  // tree blowup — so the user gets a visible heads-up before Node OOMs under
  // the exit threshold. Re-armed only after heap falls back below the floor.
  // `lastHeap < 0` marks the un-seeded first sample so a cold start that opens
  // already-high can't be mistaken for sudden growth (growth = current - last).
  let lastHeap = -1
  let warned = false
  const WARN_GROWTH_STEP = 150 * MB

  // Cooldown prevents repeated auto dumps when heap oscillates around the
  // threshold (issue #21767). `dumped` alone is not enough — it clears on
  // every transition back to `normal`.
  const cooldownRaw = process.env.HERMES_AUTO_HEAPDUMP_COOLDOWN_MS?.trim()
  const cooldownParsed = cooldownRaw ? Number(cooldownRaw) : NaN
  const cooldownMs = Number.isFinite(cooldownParsed) && cooldownParsed >= 0 ? cooldownParsed : 600_000
  let lastAutoDumpAt = 0

  const tick = async () => {
    const { heapUsed, rss } = process.memoryUsage()

    // ── STAGE 0: proactive relief in the warn regime ──────────────────────
    // The warn regime (warnBytes ≤ heap < high) is where a long render-bound
    // session lives indefinitely. Instead of only logging, ACT: prune Ink
    // caches + force GC, and if pressure persists after relief, escalate to a
    // seamless recycle via onSustainedPressure. This runs every tick the
    // session is in the warn band, not just once on the growth spike.
    if (heapUsed >= warnBytes && heapUsed < high) {
      warnPressureTicks++

      // Active relief: best-effort, non-overlapping. Prune Ink content caches
      // and force a GC pass (Node must be started with --expose-gc for
      // global.gc; the bundle is, and we no-op gracefully if not).
      if (onWarnRelief && !reliefInFlight) {
        reliefInFlight = true
        void Promise.resolve(onWarnRelief())
          .catch(() => {})
          .finally(() => {
            reliefInFlight = false
          })
      }
      forceGcIfAvailable()

      // Sustained pressure: relief didn't bring heap back under the floor for
      // `sustainedTicks` consecutive ticks → genuine render-tree blowup, ask
      // the entry to recycle the renderer. Fire once per episode.
      if (!sustainedFired && warnPressureTicks >= sustainedTicks) {
        sustainedFired = true
        onSustainedPressure?.({ heapUsed, level: 'normal', rss })
      }
    } else if (heapUsed < warnBytes) {
      // Fell back below the floor: re-arm everything for the next episode.
      warnPressureTicks = 0
      sustainedFired = false
    }

    // Sub-threshold abnormal-growth warning (existing #34095 breadcrumb). Skip
    // on the first (un-seeded) sample — we need a prior reading for the delta.
    if (heapUsed < high && lastHeap >= 0) {
      if (!warned && heapUsed >= warnBytes && heapUsed - lastHeap >= WARN_GROWTH_STEP) {
        warned = true
        onWarn?.({ heapUsed, level: 'normal', rss })
      } else if (heapUsed < warnBytes) {
        warned = false
      }
    }
    lastHeap = heapUsed

    const level: MemoryLevel = heapUsed >= critical ? 'critical' : heapUsed >= high ? 'high' : 'normal'

    if (level === 'normal') {
      dumped.clear()
      return
    }

    if (dumped.has(level) || inFlight.has(level)) {
      return
    }

    if (Date.now() - lastAutoDumpAt < cooldownMs) {
      return
    }

    inFlight.add(level)
    lastAutoDumpAt = Date.now()

    // Prune Ink content caches before dump/exit — half on 'high' (recoverable),
    // full on 'critical' (post-dump RSS reduction, keeps user running).
    // Deferred import keeps `@hermes/ink` off the cold-start critical path;
    // by the time a tick fires 10s after launch the app has already loaded
    // the same module, so this resolves instantly from the ESM cache.
    try {
      try {
        const evictInkCaches = await _ensureEvictInkCaches()
        evictInkCaches(level === 'critical' ? 'all' : 'half')
      } catch {
        // Best-effort: if the dynamic import fails for any reason we still
        // continue to the heap dump below so the user gets diagnostics.
      }

      dumped.add(level)
      const dump = await performHeapDump(level === 'critical' ? 'auto-critical' : 'auto-high').catch(() => null)
      const snap: MemorySnapshot = { heapUsed, level, rss }

      ;(level === 'critical' ? onCritical : onHigh)?.(snap, dump)
    } finally {
      inFlight.delete(level)
    }
  }

  const handle = setInterval(() => void tick(), intervalMs)

  handle.unref?.()

  return () => clearInterval(handle)
}
