import type { TodoItem } from '../types.js'

export const todoGlyph = (status: TodoItem['status']) =>
  status === 'completed' ? '[x]' : status === 'cancelled' ? '[-]' : status === 'in_progress' ? '[>]' : '[ ]'
