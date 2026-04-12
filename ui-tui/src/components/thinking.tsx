import { Box, Text } from '@hermes/ink'
import { memo, type ReactNode, useEffect, useState } from 'react'
import spinners, { type BrailleSpinnerName } from 'unicode-animations'

import { FACES, VERBS } from '../constants.js'
import {
  formatToolCall,
  parseToolTrailResultLine,
  pick,
  scaleHex,
  THINKING_COT_FADE,
  THINKING_COT_MAX,
  thinkingCotTail,
  toolTrailLabel
} from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool, ActivityItem } from '../types.js'

const THINK: BrailleSpinnerName[] = ['helix', 'breathe', 'orbit', 'dna', 'waverows', 'snake', 'pulse']
const TOOL: BrailleSpinnerName[] = ['cascade', 'scan', 'diagswipe', 'fillsweep', 'rain', 'columns', 'sparkle']

const fmtElapsed = (ms: number) => {
  const sec = Math.max(0, ms) / 1000

  return sec < 10 ? `${sec.toFixed(1)}s` : `${Math.round(sec)}s`
}

// ── Spinner ──────────────────────────────────────────────────────────

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

// ── Detail row ───────────────────────────────────────────────────────

type DetailRow = { color: string; content: ReactNode; dimColor?: boolean; key: string }

function Detail({ color, content, dimColor, t }: DetailRow & { t: Theme }) {
  return (
    <Text color={color} dimColor={dimColor}>
      <Text dimColor> └ </Text>
      {content}
    </Text>
  )
}

// ── Thinking (pre-tool fallback) ─────────────────────────────────────

export const Thinking = memo(function Thinking({ reasoning, t }: { reasoning: string; t: Theme }) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(v => v + 1), 1100)

    return () => clearInterval(id)
  }, [])

  const tail = thinkingCotTail(reasoning)
  const clipped = reasoning.length > THINKING_COT_MAX

  return (
    <Box flexDirection="column">
      <Text color={t.color.dim}>
        <Spinner color={t.color.dim} /> {FACES[tick % FACES.length] ?? '(•_•)'}{' '}
        {VERBS[tick % VERBS.length] ?? 'thinking'}…
      </Text>

      {tail ? (
        <Text wrap="truncate-end">
          {clipped &&
            Array.from({ length: Math.min(THINKING_COT_FADE, tail.length) }, (_, i) => (
              <Text color={scaleHex(t.color.dim, (i + 1) / (THINKING_COT_FADE + 1))} key={i}>
                {tail[i]}
              </Text>
            ))}

          <Text color={t.color.dim} dimColor>
            {clipped ? tail.slice(THINKING_COT_FADE) : tail}
          </Text>
        </Text>
      ) : null}
    </Box>
  )
})

// ── ToolTrail (canonical progress block) ─────────────────────────────

type Group = { color: string; content: ReactNode; details: DetailRow[]; key: string }

export const ToolTrail = memo(function ToolTrail({
  busy = false,
  reasoning = '',
  t,
  tools = [],
  trail = [],
  activity = []
}: {
  busy?: boolean
  reasoning?: string
  t: Theme
  tools?: ActiveTool[]
  trail?: string[]
  activity?: ActivityItem[]
}) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!tools.length) {
      return
    }
    const id = setInterval(() => setNow(Date.now()), 200)

    return () => clearInterval(id)
  }, [tools.length])

  if (!busy && !trail.length && !tools.length && !activity.length) {
    return null
  }

  const groups: Group[] = []
  const meta: DetailRow[] = []

  const detail = (row: DetailRow) => {
    const g = groups.at(-1)
    g ? g.details.push(row) : meta.push(row)
  }

  // ── trail → groups + details ────────────────────────────────────

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
        detail({
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
      detail({
        color: t.color.dim,
        content: groups.length ? (
          <>
            <Spinner color={t.color.amber} variant="think" /> {line}
          </>
        ) : (
          line
        ),
        dimColor: true,
        key: `tr-${i}`
      })

      continue
    }

    meta.push({ color: t.color.dim, content: line, dimColor: true, key: `tr-${i}` })
  }

  // ── live tools → groups ─────────────────────────────────────────

  for (const tool of tools) {
    groups.push({
      color: t.color.cornsilk,
      content: (
        <>
          <Spinner color={t.color.amber} variant="tool" /> {formatToolCall(tool.name, tool.context || '')}
          {tool.startedAt ? ` (${fmtElapsed(now - tool.startedAt)})` : ''}
        </>
      ),
      details: [],
      key: tool.id
    })
  }

  // ── reasoning tail → child of last group ────────────────────────

  const reasoningTail = thinkingCotTail(reasoning)

  if (groups.length && reasoningTail) {
    detail({ color: t.color.dim, content: reasoningTail, dimColor: true, key: 'cot' })
  }

  // ── activity → meta ─────────────────────────────────────────────

  for (const item of activity.slice(-4)) {
    const glyph = item.tone === 'error' ? '✗' : item.tone === 'warn' ? '!' : '·'
    const color = item.tone === 'error' ? t.color.error : item.tone === 'warn' ? t.color.warn : t.color.dim

    meta.push({ color, content: `${glyph} ${item.text}`, dimColor: item.tone === 'info', key: `a-${item.id}` })
  }

  // ── render ──────────────────────────────────────────────────────

  return (
    <Box flexDirection="column">
      {busy && !groups.length && <Thinking reasoning={reasoning} t={t} />}

      {groups.map(g => (
        <Box flexDirection="column" key={g.key}>
          <Text color={g.color}>
            <Text color={t.color.amber}>● </Text>
            {g.content}
          </Text>

          {g.details.map(d => (
            <Detail {...d} key={d.key} t={t} />
          ))}
        </Box>
      ))}

      {meta.map((row, i) => (
        <Text color={row.color} dimColor={row.dimColor} key={row.key}>
          <Text dimColor>{i === meta.length - 1 ? '└ ' : '├ '}</Text>
          {row.content}
        </Text>
      ))}
    </Box>
  )
})
