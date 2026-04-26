import { Box, Text } from '@hermes/ink'
import { memo } from 'react'

import { todoGlyph, todoTone } from '../lib/todo.js'
import type { Theme } from '../theme.js'
import type { TodoItem } from '../types.js'

const rowColor = (t: Theme, status: TodoItem['status']) => {
  const tone = todoTone(status)

  return tone === 'active' ? t.color.cornsilk : tone === 'body' ? t.color.statusFg : t.color.dim
}

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
    </Box>
  )
})
