import { Box, Text } from '@hermes/ink'
import { memo } from 'react'

import {
  aggregateKanbanActivity,
  buildKanbanActivityRows,
  collapsedActivityLabel,
  type KanbanActivityModel,
  type KanbanActivityRowModel
} from '../lib/kanbanActivity.js'
import type { Theme } from '../theme.js'

function toneColor(t: Theme, row: KanbanActivityRowModel): string {
  if (row.kind === 'board') {
    return t.color.accent
  }

  if (row.tone === 'error') {
    return t.color.error
  }

  if (row.tone === 'warning') {
    return t.color.warn
  }

  if (row.tone === 'success') {
    return t.color.ok
  }

  if (row.tone === 'accent') {
    return t.color.accent
  }

  return t.color.muted
}

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
      <Text bold color={toneColor(t, row)} wrap="truncate">
        Kanban · {row.label}
      </Text>
    )
  }

  if (row.kind === 'summary') {
    const indent = '  '.repeat(Math.max(0, row.depth - 1))

    return (
      <Text color={toneColor(t, row)} dim wrap="truncate">
        {indent}
        {row.connector} {row.label}
      </Text>
    )
  }

  const color = toneColor(t, row)

  const prefix =
    row.depth === 0
      ? `${row.rail} ${row.glyph} `
      : `${'  '.repeat(Math.max(0, row.depth - 1))}${row.connector} ${row.glyph} `

  return (
    <Box width={Math.max(1, width)}>
      <Text color={color} dim={row.tone === 'muted'} wrap="truncate">
        {prefix}
        {row.label}
      </Text>
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

  const label = collapsedActivityLabel(activity, Math.max(1, width - 2), now)
  const prefix = counts.attention > 0 ? '! ' : counts.active > 0 ? '◉ ' : counts.completed > 0 ? '● ' : '○ '

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
      <Text color={color} wrap="truncate">
        {prefix}
        {label}
      </Text>
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
