import { Text } from 'ink'
import { memo, useEffect, useState } from 'react'
import spinners, { type BrailleSpinnerName } from 'unicode-animations'

import { FACES, TOOL_VERBS, VERBS } from '../constants.js'
import type { Theme } from '../theme.js'
import type { ActiveTool } from '../types.js'

const THINK: BrailleSpinnerName[] = ['helix', 'breathe', 'orbit', 'dna', 'waverows', 'snake', 'pulse']
const TOOL: BrailleSpinnerName[] = ['cascade', 'scan', 'diagswipe', 'fillsweep', 'rain', 'columns', 'sparkle']

const pick = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)]!

export function Spinner({ color, variant = 'think' }: { color: string; variant?: 'think' | 'tool' }) {
  const [spin] = useState(() => spinners[pick(variant === 'tool' ? TOOL : THINK)])
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
  trail = []
}: {
  t: Theme
  tools?: ActiveTool[]
  trail?: string[]
}) {
  if (!trail.length && !tools.length) {
    return null
  }

  return (
    <>
      {trail.map((line, i) => (
        <Text color={line.endsWith(' ✗') ? t.color.error : t.color.dim} dimColor={!line.endsWith(' ✗')} key={`t-${i}`}>
          {t.brand.tool} {line}
        </Text>
      ))}

      {tools.map(tool => (
        <Text color={t.color.dim} key={tool.id}>
          <Spinner color={t.color.amber} variant="tool" /> {TOOL_VERBS[tool.name] ?? tool.name}
          {tool.context ? `: ${tool.context}` : ''}
        </Text>
      ))}
    </>
  )
})

export const Thinking = memo(function Thinking({ reasoning, t }: { reasoning: string; t: Theme }) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(v => v + 1), 1100)

    return () => clearInterval(id)
  }, [])

  const tail = reasoning.slice(-160).replace(/\n/g, ' ')

  return tail ? (
    <Text color={t.color.dim} dimColor wrap="truncate-end">
      💭 {tail}
    </Text>
  ) : (
    <Text color={t.color.dim}>
      <Spinner color={t.color.dim} /> {FACES[tick % FACES.length] ?? '(•_•)'} {VERBS[tick % VERBS.length] ?? 'thinking'}
      …
    </Text>
  )
})
