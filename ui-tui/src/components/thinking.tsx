import { Text } from 'ink'
import { memo, useEffect, useRef, useState } from 'react'

import { FACES, SPINNER, TOOL_VERBS, VERBS } from '../constants.js'
import { pick } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool } from '../types.js'

function SpinnerChar({ color }: { color: string }) {
  const ref = useRef(0)

  useEffect(() => {
    const id = setInterval(() => {
      ref.current = (ref.current + 1) % SPINNER.length
    }, 80)

    return () => clearInterval(id)
  }, [])

  return <Text color={color}>{SPINNER[ref.current]}</Text>
}

export const Thinking = memo(function Thinking({
  reasoning,
  t,
  thinking,
  tools
}: {
  reasoning: string
  t: Theme
  thinking?: string
  tools: ActiveTool[]
}) {
  const [verb] = useState(() => pick(VERBS))
  const [face] = useState(() => pick(FACES))

  const tail = (reasoning || thinking || '').slice(-120).replace(/\n/g, ' ')

  if (tools.length) {
    return (
      <>
        {tools.map(tool => (
          <Text color={t.color.dim} key={tool.id}>
            ⚡ {TOOL_VERBS[tool.name] ?? tool.name}…
          </Text>
        ))}
      </>
    )
  }

  if (tail) {
    return (
      <Text color={t.color.dim} dimColor wrap="truncate-end">
        💭 {tail}
      </Text>
    )
  }

  return (
    <Text color={t.color.dim}>
      <SpinnerChar color={t.color.dim} /> {face} {verb}…
    </Text>
  )
})
