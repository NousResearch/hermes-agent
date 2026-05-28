import { Box, type ScrollBoxHandle, stringWidth, Text } from '@hermes/ink'
import { type ReactNode, type RefObject, useEffect, useRef, useState } from 'react'

import { VERBS } from '../content/verbs.js'
import { fmtDuration } from '../domain/messages.js'
import { stickyPromptFromViewport } from '../domain/viewport.js'
import { fmtK } from '../lib/text.js'
import { useScrollbarSnapshot, useViewportSnapshot } from '../lib/viewportStore.js'
import type { Theme } from '../theme.js'
import type { Msg, Usage } from '../types.js'

const HEART_COLORS = ['#ff5fa2', '#ff4d6d']
const STATUS_SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

export const VERB_PAD_LEN = Math.max(...VERBS.map(verb => verb.length)) + 1
export const padVerb = (verb: string) => `${verb}…`.padEnd(VERB_PAD_LEN)

function ctxBarColor(pct: number | undefined, t: Theme) {
  if (pct == null) {
    return t.color.muted
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

export function statusRuleWidths(cols: number, cwdLabel: string) {
  const width = Math.max(1, Math.floor(cols || 1))
  const desiredSeparatorWidth = width >= 24 ? 3 : 1
  const minLeftWidth = width >= 24 ? 8 : 1
  const maxRightWidth = Math.max(0, width - desiredSeparatorWidth - minLeftWidth)

  if (!cwdLabel || maxRightWidth <= 0) {
    return { leftWidth: width, rightWidth: 0, separatorWidth: 0 }
  }

  const rightWidth = Math.max(0, Math.min(stringWidth(cwdLabel), maxRightWidth))
  const separatorWidth = rightWidth > 0 ? desiredSeparatorWidth : 0
  const leftWidth = Math.max(1, width - separatorWidth - rightWidth)

  return { leftWidth, rightWidth, separatorWidth }
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

const effortLabel = (effort?: string) => {
  const value = String(effort ?? '')
    .trim()
    .toLowerCase()

  return value && value !== 'medium' && value !== 'normal' && value !== 'default' ? value : ''
}

const shortModelLabel = (model: string) =>
  model
    .split('/')
    .pop()!
    .replace(/^claude[-_]/, '')
    .replace(/^anthropic[-_]/, '')
    .replace(/[-_]/g, ' ')
    .replace(/\b(\d+)\s+(\d+)\b/g, '$1.$2')
    .trim()

const modelLabel = (model: string, effort?: string, fast?: boolean) =>
  [shortModelLabel(model), effortLabel(effort), fast ? 'fast' : ''].filter(Boolean).join(' ')

function StatusBusyIndicator({ busy, t }: { busy: boolean; t: Theme }) {
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    if (!busy) {
      setFrame(0)

      return
    }

    const id = setInterval(() => setFrame(value => (value + 1) % STATUS_SPINNER_FRAMES.length), 120)

    return () => clearInterval(id)
  }, [busy])

  if (!busy) {
    return null
  }

  return <Text color={t.color.accent}>{STATUS_SPINNER_FRAMES[frame]} </Text>
}

export function GoodVibesHeart({ tick, t }: { tick: number; t: Theme }) {
  const [active, setActive] = useState(false)
  const [color, setColor] = useState(t.color.accent)

  useEffect(() => {
    if (tick <= 0) {
      return
    }

    const palette = [t.color.error, t.color.warn, t.color.accent]
    setColor(palette[Math.floor(Math.random() * palette.length)]!)
    setActive(true)

    const id = setTimeout(() => setActive(false), 650)

    return () => clearTimeout(id)
  }, [t.color.accent, tick])

  if (!active) {
    return null
  }

  return <Text color={color}>♥</Text>
}

export function StatusRule({
  busy,
  cols,
  model,
  modelFast,
  modelReasoningEffort,
  usage,
  sessionStartedAt,
  t
}: StatusRuleProps) {
  const pct = usage.context_percent
  const contextColor = ctxBarColor(pct, t)

  const ctxLabel = usage.context_max
    ? `${fmtK(usage.context_used ?? 0)}/${fmtK(usage.context_max)}${pct != null ? ` ${pct}%` : ''}`
    : usage.total > 0
      ? `${fmtK(usage.total)} tok`
      : ''
  const bar = usage.context_max ? ctxBar(pct) : ''

  return (
    <Box height={1}>
      <Box flexDirection="row" flexShrink={1} overflow="hidden" width={Math.max(1, cols)}>
        <Text color={t.color.border} wrap="truncate-end">
          {'─ '}
        </Text>
        <StatusBusyIndicator busy={busy} t={t} />
        <Text color={t.color.muted} wrap="truncate-end">
          {modelLabel(model, modelReasoningEffort, modelFast)}
        </Text>
        {ctxLabel ? (
          <>
            <Text color={t.color.muted} wrap="truncate-end">
              {' │ '}
            </Text>
            <Text color={contextColor} wrap="truncate-end">
              Context {ctxLabel}
            </Text>
            {bar ? (
              <Text color={contextColor} wrap="truncate-end">
                {' '}
                [{bar}]
              </Text>
            ) : null}
          </>
        ) : null}
        {sessionStartedAt ? (
          <>
            <Text color={t.color.muted}> │ </Text>
            <Text color={t.color.muted} wrap="truncate-end">
              <SessionDuration startedAt={sessionStartedAt} />
            </Text>
          </>
        ) : null}
      </Box>
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
  const { atBottom, bottom, top } = useViewportSnapshot(scrollRef)
  const text = stickyPromptFromViewport(messages, offsets, top, bottom, atBottom)

  useEffect(() => onChange(text), [onChange, text])

  return null
}

export function TranscriptScrollbar({ scrollRef, t }: TranscriptScrollbarProps) {
  const [hover, setHover] = useState(false)
  const [grab, setGrab] = useState<number | null>(null)
  const grabRef = useRef<number | null>(null)
  const { scrollHeight: total, top: pos, viewportHeight: vp } = useScrollbarSnapshot(scrollRef)

  if (!vp) {
    return <Box width={1} />
  }

  const s = scrollRef.current
  const scrollable = total > vp
  const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp
  const travel = Math.max(1, vp - thumb)
  const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0
  const thumbColor = grab !== null ? t.color.primary : hover ? t.color.accent : t.color.border
  const trackColor = hover ? t.color.border : t.color.muted

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

        grabRef.current = off
        setGrab(off)
        jump(row, off)
      }}
      onMouseDrag={(e: { localRow?: number }) =>
        jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grabRef.current ?? Math.floor(thumb / 2))
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onMouseUp={() => {
        grabRef.current = null
        setGrab(null)
      }}
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
  liveSessionCount: number
  busy: boolean
  cols: number
  cwdLabel: string
  model: string
  modelFast?: boolean
  modelReasoningEffort?: string
  sessionStartedAt?: null | number
  showCost: boolean
  status: string
  statusColor: string
  t: Theme
  turnStartedAt?: null | number
  usage: Usage
  voiceLabel?: string
  onSessionCountClick?: () => void
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
