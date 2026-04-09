import { Text } from 'ink'
import { memo, useEffect, useState } from 'react'
import spinners, { type BrailleSpinnerName } from 'unicode-animations'

import { FACES, TOOL_VERBS, VERBS } from '../constants.js'
import type { Theme } from '../theme.js'
import type { ActiveTool } from '../types.js'

const THINK_POOL: BrailleSpinnerName[] = ['helix', 'breathe', 'orbit', 'dna', 'waverows', 'snake', 'pulse']
const TOOL_POOL: BrailleSpinnerName[] = ['cascade', 'scan', 'diagswipe', 'fillsweep', 'rain', 'columns', 'sparkle']

const pick = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)]!

function Spinner({ color, variant = 'think' }: { color: string; variant?: 'think' | 'tool' }) {
  const [spin] = useState(() => spinners[pick(variant === 'tool' ? TOOL_POOL : THINK_POOL)])
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % spin.frames.length), spin.interval)

    return () => clearInterval(id)
  }, [spin])

  return <Text color={color}>{spin.frames[frame]}</Text>
}

export const Thinking = memo(function Thinking({
  reasoning,
  t,
  tools
}: {
  reasoning: string
  t: Theme
  tools: ActiveTool[]
}) {
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(v => v + 1), 1100)

    return () => clearInterval(id)
  }, [])

  const verb = VERBS[tick % VERBS.length] ?? 'thinking'
  const face = FACES[tick % FACES.length] ?? '(•_•)'
  const tail = reasoning.slice(-160).replace(/\n/g, ' ')
  const hasReasoning = !!tail

  return (
    <>
      {tools.map(tool => (
        <Text color={t.color.dim} key={tool.id}>
          <Spinner color={t.color.amber} variant="tool" /> {TOOL_VERBS[tool.name] ?? tool.name}
          {tool.context ? `: ${tool.context}` : ''}
        </Text>
      ))}

      {!tools.length && !hasReasoning && (
        <Text color={t.color.dim}>
          <Spinner color={t.color.dim} /> {face} {verb}…
        </Text>
      )}

      {tail && (
        <Text color={t.color.dim} dimColor wrap="truncate-end">
          💭 {tail}
        </Text>
      )}
    </>
  )
})
