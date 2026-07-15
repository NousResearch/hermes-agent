import { Box, Text } from '@hermes/ink'
import { memo } from 'react'

import {
  type ActivityTone,
  aggregateKanbanActivity,
  buildKanbanActivityRows,
  collapsedActivitySegments,
  type KanbanActivityModel,
  type KanbanActivityRowModel
} from '../lib/kanbanActivity.js'
import type { Theme } from '../theme.js'

// Lifecycle tones only. 'neutral' (queued work) renders in the terminal's
// default foreground so accent stays reserved for live activity.
function toneColor(t: Theme, tone: ActivityTone): string | undefined {
  if (tone === 'error') {
    return t.color.error
  }

  if (tone === 'warning') {
    return t.color.warn
  }

  if (tone === 'success') {
    return t.color.ok
  }

  if (tone === 'accent') {
    return t.color.accent
  }

  if (tone === 'muted') {
    return t.color.muted
  }

  return undefined
}

// Each signal paints its own channel: topology prefix muted, glyph and state
// word in the lifecycle tone, title in the default foreground, owner dim.
export const KanbanActivityRow = memo(function KanbanActivityRow({
  row,
  t,
  width
}: {
  row: KanbanActivityRowModel
  t: Theme
  width: number
}) {
  if (row.kind === 'board') {
    return (
      <Box width={Math.max(1, width)}>
        {row.label ? (
          <Text bold wrap="truncate">
            {row.label}
          </Text>
        ) : null}
        {row.suffix === null ? null : (
          <Text color={t.color.warn} wrap="truncate">
            {row.suffix}
          </Text>
        )}
      </Box>
    )
  }

  if (row.kind === 'summary') {
    return (
      <Text color={t.color.muted} dim wrap="truncate">
        {row.prefix}
        {row.label}
      </Text>
    )
  }

  const tone = toneColor(t, row.tone)

  return (
    <Box width={Math.max(1, width)}>
      <Text color={t.color.muted}>{row.prefix}</Text>
      <Text color={tone}>{`${row.glyph} `}</Text>
      {/* Title identity never changes: no tone, no dim, on every row kind. */}
      <Text wrap="truncate">{row.parts.title}</Text>
      {row.parts.owner === null ? null : (
        <Text color={t.color.muted} dim>{` — ${row.parts.owner}`}</Text>
      )}
      {row.parts.state === null ? null : (
        <Text color={tone}>{` · ${row.parts.state}`}</Text>
      )}
    </Box>
  )
})

export const KanbanActivityDock = memo(function KanbanActivityDock({
  activity,
  now = activity.checkedAt,
  t,
  width
}: {
  activity: KanbanActivityModel
  now?: number
  t: Theme
  width: number
}) {
  const counts = aggregateKanbanActivity(activity, now)

  if (!counts.active && !counts.attention && !counts.completed) {
    return null
  }

  // Below the medium breakpoint the shorthand badges are the lamp: each
  // segment carries its own tone ('K' neutral, '!N' warn, '◉N'/'●N' toned),
  // so no separate prefix glyph is rendered.
  if (width < 28) {
    const segments = collapsedActivitySegments(activity, Math.max(1, width), now)

    return (
      <Box height={1} width={Math.max(1, width)}>
        {segments.map((segment, index) => (
          <Text color={toneColor(t, segment.tone)} key={`${segment.text}:${index}`} wrap="truncate">
            {segment.text}
          </Text>
        ))}
      </Box>
    )
  }

  const glyph = counts.attention > 0 ? '!' : counts.active > 0 ? '◉' : '●'
  const segments = collapsedActivitySegments(activity, Math.max(1, width - 2), now, width)

  const color =
    counts.attention > 0
      ? t.color.warn
      : counts.active > 0
        ? t.color.accent
        : counts.completed > 0
          ? t.color.ok
          : t.color.muted

  return (
    <Box height={1} width={Math.max(1, width)}>
      <Text color={color}>{`${glyph} `}</Text>
      {segments.map((segment, index) => (
        <Text color={toneColor(t, segment.tone)} key={`${segment.text}:${index}`} wrap="truncate">
          {segment.text}
        </Text>
      ))}
    </Box>
  )
})

export const KanbanExecutionSpine = memo(function KanbanExecutionSpine({
  activity,
  maxChildren = 5,
  now = activity.checkedAt,
  staleAfterSeconds = 300,
  t,
  width
}: {
  activity: KanbanActivityModel
  maxChildren?: number
  now?: number
  staleAfterSeconds?: number
  t: Theme
  width: number
}) {
  const counts = aggregateKanbanActivity(activity, now)

  if (!counts.active && !counts.attention && !counts.completed) {
    return null
  }

  const rows = buildKanbanActivityRows(activity, { maxChildren, now, staleAfterSeconds, width })

  if (!rows.length) {
    return null
  }

  return (
    <Box flexDirection="column" width={Math.max(1, width)}>
      {rows.map((row, index) => (
        <KanbanActivityRow
          key={`${row.board}:${row.kind}:${row.kind === 'task' ? row.task.id : index}`}
          row={row}
          t={t}
          width={width}
        />
      ))}
    </Box>
  )
})
