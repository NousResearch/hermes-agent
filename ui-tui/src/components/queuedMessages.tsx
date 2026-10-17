import { Box, Text } from '@hermes/ink'

import { useT } from '../i18n/useT.js'
import { compactPreview } from '../lib/text.js'
import type { Theme } from '../theme.js'

export const QUEUE_WINDOW = 3

// Static braille frame for the pending-steer row. A live spinner would need a
// mounted timer in a leaf that is otherwise pure; a fixed frame keeps the
// strip deterministic (and testable) while still reading as "in flight" next
// to the plain queue rows.
export const STEER_BRAILLE = '⠋'

/**
 * Window over the queue rows. With a pending steer occupying row 1, the steer
 * row is always visible and the queue rows get QUEUE_WINDOW - 1 slots below
 * it, still centered on the row being edited.
 */
export function getQueueWindow(queueLen: number, queueEditIdx: number | null, hasSteer = false) {
  const visible = hasSteer ? Math.max(1, QUEUE_WINDOW - 1) : QUEUE_WINDOW

  const start =
    queueEditIdx === null ? 0 : Math.max(0, Math.min(queueEditIdx - 1, Math.max(0, queueLen - visible)))

  const end = Math.min(queueLen, start + visible)

  return { end, showLead: start > 0, showTail: end < queueLen, start }
}

export function QueuedMessages({
  cols,
  pendingSteer = null,
  queueEditIdx,
  queued,
  steerEditIdx = null,
  t
}: QueuedMessagesProps) {
  const T = useT().chatBits.queued
  const hasSteer = pendingSteer !== null
  const total = queued.length + (hasSteer ? 1 : 0)

  if (!total) {
    return null
  }

  const q = getQueueWindow(queued.length, queueEditIdx, hasSteer)

  const hint =
    steerEditIdx !== null
      ? T.steerEditing
      : queueEditIdx !== null
        ? T.editing(queueEditIdx + (hasSteer ? 2 : 1))
        : hasSteer
          ? T.steerHint
          : ''

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color={t.color.muted} dimColor>
        {`${hasSteer ? T.pendingHeader(total) : T.header(total)}${hint}`}
      </Text>

      {hasSteer && (
        <Text color={steerEditIdx !== null ? t.color.accent : t.color.muted} dimColor key="steer">
          {steerEditIdx !== null ? '▸' : ' '} 1. {STEER_BRAILLE} {T.steerTag}{' '}
          {compactPreview(pendingSteer!, Math.max(16, cols - 10))}
        </Text>
      )}

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
            {active ? '▸' : ' '} {idx + (hasSteer ? 2 : 1)}. {compactPreview(item, Math.max(16, cols - 10))}
          </Text>
        )
      })}

      {q.showTail && (
        <Text color={t.color.muted} dimColor>
          {'  '}
          {T.more(queued.length - q.end)}
        </Text>
      )}
    </Box>
  )
}

interface QueuedMessagesProps {
  cols: number
  /** Pending steer text accepted by the gateway; renders as the first row. */
  pendingSteer?: null | string
  queueEditIdx: number | null
  queued: string[]
  /** Non-null while the user is editing the steer row (arrowed up into it). */
  steerEditIdx?: null | number
  t: Theme
}
