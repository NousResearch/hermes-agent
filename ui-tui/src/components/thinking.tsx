import { Box, Text } from '@hermes/ink'
import { memo, type ReactNode, useEffect, useState } from 'react'
import spinners, { type BrailleSpinnerName } from 'unicode-animations'

import { FACES, VERBS } from '../constants.js'
import {
  formatToolCall,
  parseToolTrailResultLine,
  pick,
  THINKING_COT_MAX,
  thinkingPreview,
  toolTrailLabel
} from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool, ActivityItem, DetailsMode, ThinkingMode } from '../types.js'

const THINK: BrailleSpinnerName[] = ['helix', 'breathe', 'orbit', 'dna', 'waverows', 'snake', 'pulse']
const TOOL: BrailleSpinnerName[] = ['cascade', 'scan', 'diagswipe', 'fillsweep', 'rain', 'columns', 'sparkle']

const fmtElapsed = (ms: number) => {
  const sec = Math.max(0, ms) / 1000

  return sec < 10 ? `${sec.toFixed(1)}s` : `${Math.round(sec)}s`
}

// ── Primitives ───────────────────────────────────────────────────────

export function Spinner({ color, variant = 'think' }: { color: string; variant?: 'think' | 'tool' }) {
  const [spin] = useState(() => {
    const raw = spinners[pick(variant === 'tool' ? TOOL : THINK)]

    return { ...raw, frames: raw.frames.map(f => [...f][0] ?? '⠀') }
  })

  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % spin.frames.length), spin.interval)

    return () => clearInterval(id)
  }, [spin])

  return <Text color={color}>{spin.frames[frame]}</Text>
}

type DetailRow = { color: string; content: ReactNode; dimColor?: boolean; key: string }

function Detail({ color, content, dimColor }: DetailRow) {
  return (
    <Text color={color} dimColor={dimColor}>
      <Text dimColor>└ </Text>
      {content}
    </Text>
  )
}

function StreamCursor({
  color,
  dimColor,
  streaming = false,
  visible = false
}: {
  color: string
  dimColor?: boolean
  streaming?: boolean
  visible?: boolean
}) {
  const [on, setOn] = useState(true)

  useEffect(() => {
    const id = setInterval(() => setOn(v => !v), 420)

    return () => clearInterval(id)
  }, [])

  return visible ? (
    <Text color={color} dimColor={dimColor}>
      {streaming && on ? '▍' : ' '}
    </Text>
  ) : null
}

function Chevron({
  count,
  onClick,
  open,
  t,
  title,
  tone = 'dim'
}: {
  count?: number
  onClick: () => void
  open: boolean
  t: Theme
  title: string
  tone?: 'dim' | 'error' | 'warn'
}) {
  const color = tone === 'error' ? t.color.error : tone === 'warn' ? t.color.warn : t.color.dim

  return (
    <Box onClick={onClick}>
      <Text color={color} dimColor={tone === 'dim'}>
        <Text color={t.color.amber}>{open ? '▾ ' : '▸ '}</Text>
        {title}
        {typeof count === 'number' ? ` (${count})` : ''}
      </Text>
    </Box>
  )
}

// ── Thinking ─────────────────────────────────────────────────────────

export const Thinking = memo(function Thinking({
  active = false,
  mode = 'truncated',
  reasoning,
  streaming = false,
  t
}: {
  active?: boolean
  mode?: ThinkingMode
  reasoning: string
  streaming?: boolean
  t: Theme
}) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(v => v + 1), 1100)

    return () => clearInterval(id)
  }, [])

  const preview = thinkingPreview(reasoning, mode, THINKING_COT_MAX)

  return (
    <Box flexDirection="column">
      <Text color={t.color.dim}>
        <Spinner color={t.color.dim} /> {FACES[tick % FACES.length] ?? '(•_•)'}{' '}
        {VERBS[tick % VERBS.length] ?? 'thinking'}…
      </Text>

      {preview ? (
        <Text color={t.color.dim} dimColor {...(mode !== 'full' ? { wrap: 'truncate-end' as const } : {})}>
          <Text dimColor>└ </Text>
          {preview}
          <StreamCursor color={t.color.dim} dimColor streaming={streaming} visible={active} />
        </Text>
      ) : active ? (
        <Text color={t.color.dim} dimColor>
          <Text dimColor>└ </Text>
          <StreamCursor color={t.color.dim} dimColor streaming={streaming} visible={active} />
        </Text>
      ) : null}
    </Box>
  )
})

// ── ToolTrail ────────────────────────────────────────────────────────

type Group = { color: string; content: ReactNode; details: DetailRow[]; key: string }

export const ToolTrail = memo(function ToolTrail({
  busy = false,
  detailsMode = 'collapsed',
  reasoningActive = false,
  reasoning = '',
  reasoningStreaming = false,
  t,
  tools = [],
  trail = [],
  activity = []
}: {
  busy?: boolean
  detailsMode?: DetailsMode
  reasoningActive?: boolean
  reasoning?: string
  reasoningStreaming?: boolean
  t: Theme
  tools?: ActiveTool[]
  trail?: string[]
  activity?: ActivityItem[]
}) {
  const [now, setNow] = useState(() => Date.now())
  const [openThinking, setOpenThinking] = useState(false)
  const [openTools, setOpenTools] = useState(false)
  const [openMeta, setOpenMeta] = useState(false)

  useEffect(() => {
    if (!tools.length || (detailsMode === 'collapsed' && !openTools)) {
      return
    }

    const id = setInterval(() => setNow(Date.now()), 500)

    return () => clearInterval(id)
  }, [detailsMode, openTools, tools.length])

  useEffect(() => {
    if (detailsMode === 'expanded') {
      setOpenThinking(true)
      setOpenTools(true)
      setOpenMeta(true)
    }

    if (detailsMode === 'hidden') {
      setOpenThinking(false)
      setOpenTools(false)
      setOpenMeta(false)
    }
  }, [detailsMode])

  const cot = thinkingPreview(reasoning, 'full', THINKING_COT_MAX)

  if (!busy && !trail.length && !tools.length && !activity.length && !cot && !reasoningActive) {
    return null
  }

  // ── Build groups + meta ────────────────────────────────────────

  const groups: Group[] = []
  const meta: DetailRow[] = []
  const pushDetail = (row: DetailRow) => (groups.at(-1)?.details ?? meta).push(row)

  for (const [i, line] of trail.entries()) {
    const parsed = parseToolTrailResultLine(line)

    if (parsed) {
      groups.push({
        color: parsed.mark === '✗' ? t.color.error : t.color.cornsilk,
        content: parsed.detail ? parsed.call : `${parsed.call} ${parsed.mark}`,
        details: [],
        key: `tr-${i}`
      })

      if (parsed.detail) {
        pushDetail({
          color: parsed.mark === '✗' ? t.color.error : t.color.dim,
          content: parsed.detail,
          dimColor: parsed.mark !== '✗',
          key: `tr-${i}-d`
        })
      }

      continue
    }

    if (line.startsWith('drafting ')) {
      groups.push({
        color: t.color.cornsilk,
        content: toolTrailLabel(line.slice(9).replace(/…$/, '').trim()),
        details: [{ color: t.color.dim, content: 'drafting...', dimColor: true, key: `tr-${i}-d` }],
        key: `tr-${i}`
      })

      continue
    }

    if (line === 'analyzing tool output…') {
      pushDetail({
        color: t.color.dim,
        dimColor: true,
        key: `tr-${i}`,
        content: groups.length ? (
          <>
            <Spinner color={t.color.amber} variant="think" /> {line}
          </>
        ) : (
          line
        )
      })

      continue
    }

    meta.push({ color: t.color.dim, content: line, dimColor: true, key: `tr-${i}` })
  }

  for (const tool of tools) {
    groups.push({
      color: t.color.cornsilk,
      key: tool.id,
      details: [],
      content: (
        <>
          <Spinner color={t.color.amber} variant="tool" /> {formatToolCall(tool.name, tool.context || '')}
          {tool.startedAt ? ` (${fmtElapsed(now - tool.startedAt)})` : ''}
        </>
      )
    })
  }

  for (const item of activity.slice(-4)) {
    const glyph = item.tone === 'error' ? '✗' : item.tone === 'warn' ? '!' : '·'
    const color = item.tone === 'error' ? t.color.error : item.tone === 'warn' ? t.color.warn : t.color.dim
    meta.push({ color, content: `${glyph} ${item.text}`, dimColor: item.tone === 'info', key: `a-${item.id}` })
  }

  // ── Derived ────────────────────────────────────────────────────

  const hasTools = groups.length > 0
  const hasMeta = meta.length > 0
  const hasThinking = !!cot || reasoningActive || (busy && !hasTools)

  // ── Hidden: errors/warnings only ──────────────────────────────

  if (detailsMode === 'hidden') {
    const alerts = activity.filter(i => i.tone !== 'info').slice(-2)

    return alerts.length ? (
      <Box flexDirection="column">
        {alerts.map(i => (
          <Text color={i.tone === 'error' ? t.color.error : t.color.warn} key={`ha-${i.id}`}>
            {i.tone === 'error' ? '✗' : '!'} {i.text}
          </Text>
        ))}
      </Box>
    ) : null
  }

  // ── Shared render fragments ────────────────────────────────────

  const thinkingBlock = hasThinking ? (
    busy ? (
      <Thinking active={reasoningActive} mode="full" reasoning={reasoning} streaming={reasoningStreaming} t={t} />
    ) : cot ? (
      <Detail color={t.color.dim} content={cot} dimColor key="cot" />
    ) : (
      <Detail
        color={t.color.dim}
        content={<StreamCursor color={t.color.dim} dimColor streaming={reasoningStreaming} visible={reasoningActive} />}
        dimColor
        key="cot"
      />
    )
  ) : null

  const toolBlock = hasTools
    ? groups.map(g => (
        <Box flexDirection="column" key={g.key}>
          <Text color={g.color}>
            <Text color={t.color.amber}>● </Text>
            {g.content}
          </Text>
          {g.details.map(d => (
            <Detail {...d} key={d.key} />
          ))}
        </Box>
      ))
    : null

  const metaBlock = hasMeta
    ? meta.map((row, i) => (
        <Text color={row.color} dimColor={row.dimColor} key={row.key}>
          <Text dimColor>{i === meta.length - 1 ? '└ ' : '├ '}</Text>
          {row.content}
        </Text>
      ))
    : null

  // ── Expanded: flat, no accordions ──────────────────────────────

  if (detailsMode === 'expanded') {
    return (
      <Box flexDirection="column">
        {thinkingBlock}
        {toolBlock}
        {metaBlock}
      </Box>
    )
  }

  // ── Collapsed: clickable accordions ────────────────────────────

  const metaTone: 'dim' | 'error' | 'warn' = activity.some(i => i.tone === 'error')
    ? 'error'
    : activity.some(i => i.tone === 'warn')
      ? 'warn'
      : 'dim'

  return (
    <Box flexDirection="column">
      {hasThinking && (
        <>
          <Chevron onClick={() => setOpenThinking(v => !v)} open={openThinking} t={t} title="Thinking" />
          {openThinking && thinkingBlock}
        </>
      )}

      {hasTools && (
        <>
          <Chevron
            count={groups.length}
            onClick={() => setOpenTools(v => !v)}
            open={openTools}
            t={t}
            title="Tool calls"
          />
          {openTools && toolBlock}
        </>
      )}

      {hasMeta && (
        <>
          <Chevron
            count={meta.length}
            onClick={() => setOpenMeta(v => !v)}
            open={openMeta}
            t={t}
            title="Activity"
            tone={metaTone}
          />
          {openMeta && metaBlock}
        </>
      )}
    </Box>
  )
})
