import { Box, type ScrollBoxHandle, Text, stringWidth } from '@hermes/ink'
import { type ReactNode, type RefObject, useCallback, useEffect, useState, useSyncExternalStore } from 'react'

import { FACES } from '../content/faces.js'
import { VERBS } from '../content/verbs.js'
import { fmtDuration } from '../domain/messages.js'
import { stickyPromptFromViewport } from '../domain/viewport.js'
import { fmtK } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { Msg, Usage } from '../types.js'

const FACE_TICK_MS = 2500
const HEART_COLORS = ['#ff5fa2', '#ff4d6d']
const STATUS_RULE_DIVIDER_WIDTH = 3
const STATUS_RULE_MIN_LEFT_WIDTH = 12
const STATUS_RULE_MAX_RIGHT_SHARE = 0.4

function FaceTicker({ color }: { color: string }) {
  const [tick, setTick] = useState(() => Math.floor(Math.random() * 1000))

  useEffect(() => {
    const id = setInterval(() => setTick(n => n + 1), FACE_TICK_MS)

    return () => clearInterval(id)
  }, [])

  return (
    <Text color={color}>
      {FACES[tick % FACES.length]} {VERBS[tick % VERBS.length]}…
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

export function nextSessionDurationDelay(elapsedMs: number) {
  const remainder = elapsedMs % 1000
  const nextSecond = remainder === 0 ? 1000 : 1000 - remainder

  if (elapsedMs < 3600_000) {
    return nextSecond
  }

  const elapsedThisMinuteMs = elapsedMs % 60_000
  return elapsedThisMinuteMs === 0 ? 60_000 : 60_000 - elapsedThisMinuteMs
}

export function getStatusRuleLayout(cols: number, cwdLabel: string) {
  const totalCols = Math.max(1, cols)
  const reserved = Math.min(totalCols, STATUS_RULE_DIVIDER_WIDTH + STATUS_RULE_MIN_LEFT_WIDTH)
  const remaining = Math.max(0, totalCols - reserved)
  const desiredRight = Math.min(stringWidth(cwdLabel), Math.max(0, Math.floor(totalCols * STATUS_RULE_MAX_RIGHT_SHARE)))
  const rightWidth = Math.min(remaining, desiredRight)
  const dividerWidth = rightWidth > 0 ? Math.min(STATUS_RULE_DIVIDER_WIDTH, Math.max(0, totalCols - rightWidth)) : 0
  const leftWidth = Math.max(1, totalCols - rightWidth - dividerWidth)

  return { dividerWidth, leftWidth, rightWidth }
}

function SessionDuration({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout> | null = null
    let cancelled = false

    const schedule = () => {
      const elapsed = Date.now() - startedAt
      timeout = setTimeout(() => {
        if (cancelled) {
          return
        }

        setNow(Date.now())
        schedule()
      }, nextSessionDurationDelay(elapsed))
    }

    setNow(Date.now())
    schedule()

    return () => {
      cancelled = true
      if (timeout) {
        clearTimeout(timeout)
      }
    }
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

  return <Text color={color}>{active ? '♥' : ' '}</Text>
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
  const { dividerWidth, leftWidth, rightWidth } = getStatusRuleLayout(cols, cwdLabel)

  return (
    <Box width={Math.max(1, cols)}>
      <Box flexShrink={1} width={leftWidth}>
        <Text color={t.color.bronze} wrap="truncate-end">
          {'─ '}
          {busy ? <FaceTicker color={statusColor} /> : <Text color={statusColor}>{status}</Text>}
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
          {voiceLabel ? <Text color={t.color.dim}> │ {voiceLabel}</Text> : null}
          {bgCount > 0 ? <Text color={t.color.dim}> │ {bgCount} bg</Text> : null}
          {showCost && typeof usage.cost_usd === 'number' ? (
            <Text color={t.color.dim}> │ ${usage.cost_usd.toFixed(4)}</Text>
          ) : null}
        </Text>
      </Box>

      {dividerWidth > 0 ? (
        <Box flexShrink={0} width={dividerWidth}>
          <Text color={t.color.bronze}> ─ </Text>
        </Box>
      ) : null}

      {rightWidth > 0 ? (
        <Box flexShrink={1} width={rightWidth}>
          <Text color={t.color.label} wrap="truncate-start">
            {cwdLabel}
          </Text>
        </Box>
      ) : null}
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
      const s = scrollRef.current

      if (!s) {
        return NaN
      }

      const top = Math.max(0, s.getScrollTop() + s.getPendingDelta())

      return s.isSticky() ? -1 - top : top
    },
    () => NaN
  )

  const s = scrollRef.current
  const top = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0))
  const text = stickyPromptFromViewport(messages, offsets, top, s?.isSticky() ?? true)

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
  sessionStartedAt?: number | null
  showCost: boolean
  status: string
  statusColor: string
  t: Theme
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
