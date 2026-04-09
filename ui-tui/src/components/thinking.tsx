import { Text } from 'ink'
import { memo, useEffect, useState } from 'react'
import spinners, { type BrailleSpinnerName } from 'unicode-animations'

import { FACES, TOOL_VERBS, VERBS } from '../constants.js'
import {
  isToolTrailResultLine,
  lastCotTrailIndex,
  pick,
  scaleHex,
  THINKING_COT_FADE,
  THINKING_COT_MAX,
  thinkingCotTail
} from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool, ActivityItem } from '../types.js'

const THINK: BrailleSpinnerName[] = ['helix', 'breathe', 'orbit', 'dna', 'waverows', 'snake', 'pulse']
const TOOL: BrailleSpinnerName[] = ['cascade', 'scan', 'diagswipe', 'fillsweep', 'rain', 'columns', 'sparkle']

const tone = (item: ActivityItem, t: Theme) =>
  item.tone === 'error' ? t.color.error : item.tone === 'warn' ? t.color.warn : t.color.dim

const activityGlyph = (item: ActivityItem) => (item.tone === 'error' ? '✗' : item.tone === 'warn' ? '⚠' : '·')

const TreeFork = ({ last }: { last: boolean }) => <Text dimColor>{last ? '└─ ' : '├─ '}</Text>

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

export const ToolTrail = memo(function ToolTrail({
  t,
  tools = [],
  trail = [],
  activity = [],
  animateCot = false
}: {
  t: Theme
  tools?: ActiveTool[]
  trail?: string[]
  activity?: ActivityItem[]
  animateCot?: boolean
}) {
  if (!trail.length && !tools.length && !activity.length) {
    return null
  }

  const act = activity.slice(-4)
  const rowCount = trail.length + tools.length + act.length
  const activeCotIdx = animateCot && !tools.length ? lastCotTrailIndex(trail) : -1

  return (
    <>
      {trail.map((line, i) => {
        const lastInBlock = i === rowCount - 1

        if (isToolTrailResultLine(line)) {
          return (
            <Text
              color={line.endsWith(' ✗') ? t.color.error : t.color.dim}
              dimColor={!line.endsWith(' ✗')}
              key={`t-${i}`}
            >
              <TreeFork last={lastInBlock} />
              {line}
            </Text>
          )
        }

        if (i === activeCotIdx) {
          return (
            <Text color={t.color.dim} key={`c-${i}`}>
              <TreeFork last={lastInBlock} />
              <Spinner color={t.color.amber} variant="think" /> {line}
            </Text>
          )
        }

        return (
          <Text color={t.color.dim} dimColor key={`c-${i}`}>
            <TreeFork last={lastInBlock} />
            {line}
          </Text>
        )
      })}

      {tools.map((tool, j) => {
        const lastInBlock = trail.length + j === rowCount - 1

        return (
          <Text color={t.color.dim} key={tool.id}>
            <TreeFork last={lastInBlock} />
            <Spinner color={t.color.amber} variant="tool" /> {TOOL_VERBS[tool.name] ?? tool.name}
            {tool.context ? `: ${tool.context}` : ''}
          </Text>
        )
      })}

      {act.map((item, k) => {
        const lastInBlock = trail.length + tools.length + k === rowCount - 1

        return (
          <Text color={tone(item, t)} dimColor={item.tone === 'info'} key={`a-${item.id}`}>
            <TreeFork last={lastInBlock} />
            {activityGlyph(item)} {item.text}
          </Text>
        )
      })}
    </>
  )
})

export const Thinking = memo(function Thinking({ reasoning, t }: { reasoning: string; t: Theme }) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(v => v + 1), 1100)

    return () => clearInterval(id)
  }, [])

  const tail = thinkingCotTail(reasoning)
  const clipped = reasoning.length > THINKING_COT_MAX

  return (
    <>
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
    </>
  )
})
