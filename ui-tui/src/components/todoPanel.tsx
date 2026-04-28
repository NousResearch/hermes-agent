import { Box, Text } from '@hermes/ink'
import { memo, useState } from 'react'

import { countPendingTodos } from '../lib/liveProgress.js'
import { todoGlyph, todoTone } from '../lib/todo.js'
import type { Theme } from '../theme.js'
import type { TodoItem } from '../types.js'

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

  const done = todos.filter(todo => todo.status === 'completed').length
  const pending = countPendingTodos(todos)

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box onClick={handleToggle}>
        <Text color={t.color.muted}>
          <Text color={t.color.accent}>{effectiveCollapsed ? '▸ ' : '▾ '}</Text>
          <Text bold color={t.color.text}>
            Todo
          </Text>{' '}
          <Text color={t.color.statusFg} dim>
            ({done}/{todos.length})
          </Text>
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
          {todos.map(todo => {
            const tone = todoTone(todo.status)
            const color = rowColor(t, todo.status)

            return (
              <Text color={color} dim={tone === 'dim'} key={todo.id}>
                <Text color={color}>{todoGlyph(todo.status)} </Text>
                {todo.content}
              </Text>
            )
          })}
        </Box>
      )}
    </Box>
  )
})
