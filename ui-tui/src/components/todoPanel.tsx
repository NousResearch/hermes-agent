import { Box, Text } from '@hermes/ink'
import { memo, useState } from 'react'

import { countPendingTodos } from '../lib/liveProgress.js'
import { stripAnsi } from '@hermes/shared/ansi'
import { todoGlyph, todoTone, todoTree } from '../lib/todo.js'
import type { Theme } from '../theme.js'
import type { TodoItem } from '../types.js'

const MAX_VISIBLE_TODOS = 7
const displayContent = (content: string) => stripAnsi(content).trim().replace(/\s+/g, ' ')

const todoWindow = (todos: TodoItem[]) => {
  if (todos.length <= MAX_VISIBLE_TODOS) {
    return { hidden: 0, rows: todos }
  }

  const runningIndex = todos.findIndex(todo => todo.status === 'in_progress')
  const pendingIndex = todos.findIndex(todo => todo.status === 'pending')
  const anchor = Math.max(0, runningIndex >= 0 ? runningIndex : pendingIndex)
  let start = Math.max(0, anchor - 1)
  const end = Math.min(todos.length, start + MAX_VISIBLE_TODOS)
  start = Math.max(0, end - MAX_VISIBLE_TODOS)

  return { hidden: todos.length - (end - start), rows: todos.slice(start, end) }
}

const rowColor = (t: Theme, status: TodoItem['status']) => {
  const tone = todoTone(status)

  return tone === 'active' ? t.color.text : tone === 'body' ? t.color.statusFg : t.color.muted
}

export const TodoPanel = memo(function TodoPanel({
  collapsed,
  defaultCollapsed = false,
  incomplete = false,
  onToggle,
  t,
  todos
}: {
  collapsed?: boolean
  defaultCollapsed?: boolean
  incomplete?: boolean
  onToggle?: () => void
  t: Theme
  todos: TodoItem[]
}) {
  // Fallback local state for archived todos in transcript where there's no
  // external controller. Live TodoPanel passes collapsed+onToggle from the
  // turn store so clicks still work there.
  const [localCollapsed, setLocalCollapsed] = useState(defaultCollapsed)
  const isControlled = typeof collapsed === 'boolean'
  const effectiveCollapsed = isControlled ? collapsed : localCollapsed

  const handleToggle = () => {
    if (onToggle) {
      onToggle()

      return
    }

    if (!isControlled) {
      setLocalCollapsed(v => !v)
    }
  }

  if (!todos.length) {
    return null
  }

  const counted = todos.filter(todo => todo.status !== 'cancelled')
  const done = counted.filter(todo => todo.status === 'completed').length
  const pending = countPendingTodos(todos)
  const current =
    todos.find(todo => todo.status === 'in_progress') ??
    todos.find(todo => todo.status === 'pending') ??
    todos.at(-1)
  const window = todoWindow(todos)

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box onClick={handleToggle}>
        <Text color={t.color.muted} wrap="truncate-end">
          <Text color={t.color.accent}>{effectiveCollapsed ? '▸ ' : '▾ '}</Text>
          <Text bold color={t.color.text}>
            Todo
          </Text>{' '}
          <Text color={t.color.statusFg} dim>
            ({done}/{counted.length})
          </Text>
          {effectiveCollapsed && current && (
            <Text color={rowColor(t, current.status)}>
              {' · '}
              {todoGlyph(current.status)} {displayContent(current.content)}
              {onToggle ? ' · Ctrl+T' : ''}
            </Text>
          )}
          {incomplete && pending > 0 && (
            <Text color={t.color.muted} dim>
              {' '}
              · incomplete · {pending} still {pending === 1 ? 'pending' : 'pending/in_progress'}
            </Text>
          )}
        </Text>
      </Box>

      {!effectiveCollapsed && (
        <Box flexDirection="column" marginLeft={2}>
          {todoTree(todos).map(([todo, depth]) => {
            const tone = todoTone(todo.status)
            const color = rowColor(t, todo.status)

            return (
              <Box key={todo.id} marginLeft={Math.min(depth, 4) * 2}>
                <Text color={color} dim={tone === 'dim'}>
                  <Text color={color}>{todoGlyph(todo.status)} </Text>
                  {todo.content}
                </Text>
              </Box>
            )
          })}
          {window.hidden > 0 && (
            <Text color={t.color.muted} dim>
              … +{window.hidden} more
            </Text>
          )}
        </Box>
      )}
    </Box>
  )
})
