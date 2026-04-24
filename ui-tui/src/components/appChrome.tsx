import { Box, type ScrollBoxHandle, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type ReactNode, type RefObject, useCallback, useEffect, useMemo, useState, useSyncExternalStore } from 'react'

import { $delegationState } from '../app/delegationStore.js'
import { $turnState } from '../app/turnStore.js'
import { FACES } from '../content/faces.js'
import { VERBS } from '../content/verbs.js'
import { fmtDuration } from '../domain/messages.js'
import { stickyPromptFromViewport } from '../domain/viewport.js'
import { buildSubagentTree, treeTotals, widthByDepth } from '../lib/subagentTree.js'
import { fmtK } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { Msg, Usage } from '../types.js'

const FACE_TICK_MS = 2500
const HEART_COLORS = ['#ff5fa2', '#ff4d6d']

function FaceTicker({ color, startedAt }: { color: string; startedAt?: null | number }) {
  const [tick, setTick] = useState(() => Math.floor(Math.random() * 1000))
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const face = setInterval(() => setTick(n => n + 1), FACE_TICK_MS)
    const clock = setInterval(() => setNow(Date.now()), 1000)

    return () => {
      clearInterval(face)
      clearInterval(clock)
    }
  }, [])

  return (
    <Text color={color}>
      {FACES[tick % FACES.length]} {VERBS[tick % VERBS.length]}…{startedAt ? ` · ${fmtDuration(now - startedAt)}` : ''}
    </Text>
  )
}

function ctxBarColor(pct: number | undefined, t: Theme) {
  if (pct == null) {
    return t.color.dim
  }

  if (pct >= 95) {
    return t.color.statusCritical
  }

  if (pct > 80) {
    return t.color.statusBad
  }

  if (pct >= 50) {
    return t.color.statusWarn
  }

  return t.color.statusGood
}

function ctxBar(pct: number | undefined, w = 10) {
  const p = Math.max(0, Math.min(100, pct ?? 0))
  const filled = Math.round((p / 100) * w)

  return '█'.repeat(filled) + '░'.repeat(w - filled)
}

function SpawnHud({ t }: { t: Theme }) {
  // Tight HUD that only appears when the session is actually fanning out.
  // Colour escalates to warn/error as depth or concurrency approaches the cap.
  const delegation = useStore($delegationState)
  const turn = useStore($turnState)

  const tree = useMemo(() => buildSubagentTree(turn.subagents), [turn.subagents])
  const totals = useMemo(() => treeTotals(tree), [tree])

  if (!totals.descendantCount && !delegation.paused) {
    return null
  }

  const maxDepth = delegation.maxSpawnDepth
  const maxConc = delegation.maxConcurrentChildren
  const depth = Math.max(0, totals.maxDepthFromHere)
  const active = totals.activeCount

  // `max_concurrent_children` is a per-parent cap, not a global one.
  // `activeCount` sums every running agent across the tree and would
  // over-warn for multi-orchestrator runs.  The widest level of the tree
  // is a closer proxy to "most concurrent spawns that could be hitting a
  // single parent's slot budget".
  const widestLevel = widthByDepth(tree).reduce((a, b) => Math.max(a, b), 0)
  const depthRatio = maxDepth ? depth / maxDepth : 0
  const concRatio = maxConc ? widestLevel / maxConc : 0
  const ratio = Math.max(depthRatio, concRatio)

  const color = delegation.paused || ratio >= 1 ? t.color.error : ratio >= 0.66 ? t.color.warn : t.color.dim

  const pieces: string[] = []

  if (delegation.paused) {
    pieces.push('⏸ paused')
  }

  if (totals.descendantCount > 0) {
    const depthLabel = maxDepth ? `${depth}/${maxDepth}` : `${depth}`
    pieces.push(`d${depthLabel}`)

    if (active > 0) {
      // Label pairs the widest-level count (drives concRatio above) with
      // the total active count for context.  `W/cap` triggers the warn,
      // `+N` is everything else currently running across the tree.
      const extra = Math.max(0, active - widestLevel)
      const widthLabel = maxConc ? `${widestLevel}/${maxConc}` : `${widestLevel}`
      const suffix = extra > 0 ? `+${extra}` : ''
      pieces.push(`⚡${widthLabel}${suffix}`)
    }
  }

  const atCap = depthRatio >= 1 || concRatio >= 1

  return (
    <Text color={color}>
      {atCap ? ' │ ⚠ ' : ' │ '}
      {pieces.join(' ')}
    </Text>
  )
}

function SessionDuration({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 1000)

    return () => clearInterval(id)
  }, [startedAt])

  return fmtDuration(now - startedAt)
}

export function GoodVibesHeart({ tick, t }: { tick: number; t: Theme }) {
  const [active, setActive] = useState(false)
  const [color, setColor] = useState(t.color.amber)

  useEffect(() => {
    if (tick <= 0) {
      return
    }

    const palette = [...HEART_COLORS, t.color.amber]
    setColor(palette[Math.floor(Math.random() * palette.length)]!)
    setActive(true)

    const id = setTimeout(() => setActive(false), 650)

    return () => clearTimeout(id)
  }, [t.color.amber, tick])

  if (!active) {
    return null
  }

  return <Text color={color}>♥</Text>
}

export function StatusRule({
  cwdLabel,
  cols,
  busy,
  status,
  statusColor,
  model,
  usage,
  bgCount,
  sessionStartedAt,
  showCost,
  turnStartedAt,
  voiceLabel,
  t
}: StatusRuleProps) {
  const pct = usage.context_percent
  const barColor = ctxBarColor(pct, t)

  const ctxLabel = usage.context_max
    ? `${fmtK(usage.context_used ?? 0)}/${fmtK(usage.context_max)}`
    : usage.total > 0
      ? `${fmtK(usage.total)} tok`
      : ''

  const bar = usage.context_max ? ctxBar(pct) : ''
  const leftWidth = Math.max(12, cols - cwdLabel.length - 3)

  return (
    <Box height={1}>
      <Box flexShrink={1} width={leftWidth}>
        <Text color={t.color.bronze} wrap="truncate-end">
          {'─ '}
          {busy ? (
            <FaceTicker color={statusColor} startedAt={turnStartedAt} />
          ) : (
            <Text color={statusColor}>{status}</Text>
          )}
          <Text color={t.color.dim}> │ {model}</Text>
          {ctxLabel ? <Text color={t.color.dim}> │ {ctxLabel}</Text> : null}
          {bar ? (
            <Text color={t.color.dim}>
              {' │ '}
              <Text color={barColor}>[{bar}]</Text> <Text color={barColor}>{pct != null ? `${pct}%` : ''}</Text>
            </Text>
          ) : null}
          {sessionStartedAt ? (
            <Text color={t.color.dim}>
              {' │ '}
              <SessionDuration startedAt={sessionStartedAt} />
            </Text>
          ) : null}
          <SpawnHud t={t} />
          {voiceLabel ? (
            <Text
              color={
                voiceLabel.startsWith('●')
                  ? t.color.error
                  : voiceLabel.startsWith('◉')
                    ? t.color.warn
                    : t.color.dim
              }
            >
              {' │ '}
              {voiceLabel}
            </Text>
          ) : null}
          {bgCount > 0 ? <Text color={t.color.dim}> │ {bgCount} bg</Text> : null}
          {showCost && typeof usage.cost_usd === 'number' ? (
            <Text color={t.color.dim}> │ ${usage.cost_usd.toFixed(4)}</Text>
          ) : null}
        </Text>
      </Box>

      <Text color={t.color.bronze}> ─ </Text>
      <Text color={t.color.label}>{cwdLabel}</Text>
    </Box>
  )
}

export function FloatBox({ children, color }: { children: ReactNode; color: string }) {
  return (
    <Box
      alignSelf="flex-start"
      borderColor={color}
      borderStyle="double"
      flexDirection="column"
      marginTop={1}
      opaque
      paddingX={1}
    >
      {children}
    </Box>
  )
}

export function StickyPromptTracker({ messages, offsets, scrollRef, onChange }: StickyPromptTrackerProps) {
  useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => {}), [scrollRef]),
    () => {
      const { atBottom, top } = getStickyViewport(scrollRef.current)

      return atBottom ? -1 - top : top
    },
    () => NaN
  )

  const { atBottom, bottom, top } = getStickyViewport(scrollRef.current)
  const text = stickyPromptFromViewport(messages, offsets, top, bottom, atBottom)

  useEffect(() => onChange(text), [onChange, text])

  return null
}

export function TranscriptScrollbar({ scrollRef, t }: TranscriptScrollbarProps) {
  useSyncExternalStore(
    useCallback((cb: () => void) => scrollRef.current?.subscribe(cb) ?? (() => {}), [scrollRef]),
    () => {
      const s = scrollRef.current

      if (!s) {
        return NaN
      }

      const vp = Math.max(0, s.getViewportHeight())
      const total = Math.max(vp, s.getScrollHeight())
      const top = Math.max(0, s.getScrollTop() + s.getPendingDelta())
      const thumb = total > vp ? Math.max(1, Math.round((vp * vp) / total)) : vp
      const travel = Math.max(1, vp - thumb)
      const thumbTop = total > vp ? Math.round((top / Math.max(1, total - vp)) * travel) : 0

      return `${thumbTop}:${thumb}:${vp}`
    },
    () => ''
  )

  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<number | null>(null)

  const s = scrollRef.current
  const vp = Math.max(0, s?.getViewportHeight() ?? 0)

  if (!vp) {
    return <Box width={1} />
  }

  const total = Math.max(vp, s?.getScrollHeight() ?? vp)
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const pos = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0
  const thumbColor = grab !== null ? t.color.gold : hover ? t.color.amber : t.color.bronze
  const trackColor = hover ? t.color.bronze : t.color.dim

  const jump = (row: number, offset: number) => {
    if (!s || !scrollable) {
      return
    }

    s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)))
  }

  return (
    <Box
      flexDirection="column"
      onMouseDown={(e: { localRow?: number }) => {
        const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0))
        const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2)
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grab ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => setGrab(null)}
      width={1}
    >
      {!scrollable ? (
        <Text color={trackColor} dim>
          {' \n'.repeat(Math.max(0, vp - 1))}{' '}
        </Text>
      ) : (
        <>
          {thumbTop > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {`${'│\n'.repeat(Math.max(0, thumbTop - 1))}${thumbTop > 0 ? '│' : ''}`}
            </Text>
          ) : null}
          {thumb > 0 ? (
            <Text color={thumbColor}>{`${'┃\n'.repeat(Math.max(0, thumb - 1))}${thumb > 0 ? '┃' : ''}`}</Text>
          ) : null}
          {vp - thumbTop - thumb > 0 ? (
            <Text color={trackColor} dim={!hover}>
              {`${'│\n'.repeat(Math.max(0, vp - thumbTop - thumb - 1))}${vp - thumbTop - thumb > 0 ? '│' : ''}`}
            </Text>
          ) : null}
        </>
      )}
    </Box>
  )
}

interface StatusRuleProps {
  bgCount: number
  busy: boolean
  cols: number
  cwdLabel: string
  model: string
  sessionStartedAt?: null | number
  showCost: boolean
  status: string
  statusColor: string
  t: Theme
  turnStartedAt?: null | number
  usage: Usage
  voiceLabel?: string
}

interface StickyPromptTrackerProps {
  messages: readonly Msg[]
  offsets: ArrayLike<number>
  onChange: (text: string) => void
  scrollRef: RefObject<ScrollBoxHandle | null>
}

interface TranscriptScrollbarProps {
  scrollRef: RefObject<ScrollBoxHandle | null>
  t: Theme
}

function getStickyViewport(s?: ScrollBoxHandle | null) {
  const top = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))
  const vp = Math.max(0, s?.getViewportHeight() ?? 0)
  const total = Math.max(vp, s?.getScrollHeight() ?? vp)

  return {
    atBottom: (s?.isSticky() ?? true) || top + vp >= total - 2,
    bottom: top + vp,
    top
  }
}
