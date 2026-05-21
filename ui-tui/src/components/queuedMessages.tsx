import { Box, Text } from '@hermes/ink'

import { useI18n } from '../i18n/index.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

export const QUEUE_WINDOW = 3

export function getQueueWindow(queueLen: number, queueEditIdx: number | null) {
  const start =
    queueEditIdx === null ? 0 : Math.max(0, Math.min(queueEditIdx - 1, Math.max(0, queueLen - QUEUE_WINDOW)))

  const end = Math.min(queueLen, start + QUEUE_WINDOW)

  return { end, showLead: start > 0, showTail: end < queueLen, start }
}

export function QueuedMessages({ cols, queueEditIdx, queued, t }: QueuedMessagesProps) {
  const { t: ti } = useI18n()
  if (!queued.length) {
    return null
  }

  const q = getQueueWindow(queued.length, queueEditIdx)

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color={t.color.muted} dimColor>
        {ti('queue.header', { count: String(queued.length) })}
        {queueEditIdx !== null ? ti('queue.editHint', { idx: String(queueEditIdx + 1) }) : ''}
      </Text>

      {q.showLead && (
        <Text color={t.color.muted} dimColor>
          {' '}
          …
        </Text>
      )}

      {queued.slice(q.start, q.end).map((item, i) => {
        const idx = q.start + i
        const active = queueEditIdx === idx

        return (
          <Text color={active ? t.color.accent : t.color.muted} dimColor key={`${idx}-${item.slice(0, 16)}`}>
            {active ? '▸' : ' '} {idx + 1}. {compactPreview(item, Math.max(16, cols - 10))}
          </Text>
        )
      })}

      {q.showTail && (
        <Text color={t.color.muted} dimColor>
          {'  '}{ti('queue.more', { count: String(queued.length - q.end) })}
        </Text>
      )}
    </Box>
  )
}

interface QueuedMessagesProps {
  cols: number
  queueEditIdx: number | null
  queued: string[]
  t: Theme
}
