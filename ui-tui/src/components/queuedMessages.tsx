import { Box, Text } from 'ink'

import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

export function QueuedMessages({
  cols,
  queueEditIdx,
  queued,
  t
}: {
  cols: number
  queueEditIdx: number | null
  queued: string[]
  t: Theme
}) {
  if (!queued.length) {
    return null
  }

  const qWindow = 3
  const qStart = queueEditIdx === null ? 0 : Math.max(0, Math.min(queueEditIdx - 1, queued.length - qWindow))
  const qEnd = Math.min(queued.length, qStart + qWindow)

  return (
    <Box flexDirection="column">
      <Text color={t.color.dim} dimColor>
        queued ({queued.length}){queueEditIdx !== null ? ` · editing ${queueEditIdx + 1}` : ''}
      </Text>
      {qStart > 0 && (
        <Text color={t.color.dim} dimColor>
          {' '}
          …
        </Text>
      )}
      {queued.slice(qStart, qEnd).map((item, i) => {
        const idx = qStart + i
        const active = queueEditIdx === idx

        return (
          <Text color={active ? t.color.amber : t.color.dim} dimColor key={`${idx}-${item.slice(0, 16)}`}>
            {active ? '▸' : ' '} {idx + 1}. {compactPreview(item, Math.max(16, cols - 10))}
          </Text>
        )
      })}
      {qEnd < queued.length && (
        <Text color={t.color.dim} dimColor>
          {'  '}…and {queued.length - qEnd} more
        </Text>
      )}
    </Box>
  )
}
