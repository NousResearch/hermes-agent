import { Box, Text } from 'ink'
import { useEffect, useState } from 'react'

import { FACES, SPINNER, TOOL_VERBS, VERBS } from '../constants.js'
import { pick } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool } from '../types.js'

export function Thinking({
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
  const [frame, setFrame] = useState(0)
  const [verb] = useState(() => pick(VERBS))
  const [face] = useState(() => pick(FACES))

  useEffect(() => {
    const id = setInterval(() => setFrame(f => (f + 1) % SPINNER.length), 80)

    return () => clearInterval(id)
  }, [])

  const tail = (reasoning || thinking || '').slice(-120).replace(/\n/g, ' ')

  return (
    <Box flexDirection="column">
      {tools.length ? (
        tools.map(tool => (
          <Text color={t.color.dim} key={tool.id}>
            {SPINNER[frame]} {TOOL_VERBS[tool.name] ?? '⚡ ' + tool.name}…
          </Text>
        ))
      ) : tail ? (
        <Text color={t.color.dim} dimColor wrap="truncate-end">
          {SPINNER[frame]} 💭 {tail}
        </Text>
      ) : (
        <Text color={t.color.dim}>
          {SPINNER[frame]} {face} {verb}…
        </Text>
      )}
    </Box>
  )
}
