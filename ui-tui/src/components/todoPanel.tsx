import { Box, Text } from '@hermes/ink'
import { memo } from 'react'

import { todoGlyph } from '../lib/todo.js'
import type { Theme } from '../theme.js'
import type { TodoItem } from '../types.js'

export const TodoPanel = memo(function TodoPanel({ t, todos }: { t: Theme; todos: TodoItem[] }) {
  if (!todos.length) {
    return null
  }

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color={t.color.dim}>
        <Text color={t.color.amber}>▾ </Text>
        <Text bold color={t.color.cornsilk}>
          Todo
        </Text>{' '}
        <Text color={t.color.statusFg} dim>
          ({todos.filter(todo => todo.status === 'completed').length}/{todos.length})
        </Text>
      </Text>
      <Box flexDirection="column" marginLeft={2}>
        {todos.map(todo => {
          const done = todo.status === 'completed'
          const cancel = todo.status === 'cancelled'
          const active = todo.status === 'in_progress'

          return (
            <Text
              color={done || cancel ? t.color.dim : active ? t.color.cornsilk : t.color.statusFg}
              dim={done || cancel}
              key={todo.id}
            >
              <Text color={active ? t.color.amber : done ? t.color.ok : cancel ? t.color.error : t.color.dim}>
                {todoGlyph(todo.status)}{' '}
              </Text>
              {todo.content}
            </Text>
          )
        })}
      </Box>
    </Box>
  )
})
