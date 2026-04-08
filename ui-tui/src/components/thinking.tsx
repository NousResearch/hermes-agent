import { Text } from 'ink'
import { memo, useEffect, useState } from 'react'

import { FACES, SPINNER, TOOL_VERBS, VERBS } from '../constants.js'
import { pick } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool } from '../types.js'

function Spinner({ color }: { color: string }) {
  const [i, setI] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setI(p => (p + 1) % SPINNER.length), 80)
    return () => clearInterval(id)
  }, [])

  return <Text color={color}>{SPINNER[i]}</Text>
}

export const Thinking = memo(function Thinking({
  reasoning, t, tools
}: {
  reasoning: string; t: Theme; tools: ActiveTool[]
}) {
  const [verb, setVerb] = useState(() => pick(VERBS))
  const [face, setFace] = useState(() => pick(FACES))

  useEffect(() => {
    const id = setInterval(() => { setVerb(pick(VERBS)); setFace(pick(FACES)) }, 1100)
    return () => clearInterval(id)
  }, [])

  const tail = reasoning.slice(-160).replace(/\n/g, ' ')

  return (
    <>
      {tools.map(tool => (
        <Text color={t.color.dim} key={tool.id}>
          <Spinner color={t.color.amber} /> {TOOL_VERBS[tool.name] ?? tool.name}
          {tool.context ? `  ${tool.context}` : ''}
        </Text>
      ))}

      {!tools.length && (
        <Text color={t.color.dim}>
          <Spinner color={t.color.dim} /> {face} {verb}…
        </Text>
      )}

      {tail && <Text color={t.color.dim} dimColor wrap="truncate-end">💭 {tail}</Text>}
    </>
  )
})
