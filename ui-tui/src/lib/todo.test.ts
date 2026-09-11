import { describe, expect, it } from 'vitest'

import { todoGlyph, todoTone, todoTree } from './todo.js'

describe('todoGlyph', () => {
  it('uses fixed-width ASCII markers so the active row does not render wide or emoji-like', () => {
    expect(todoGlyph('completed')).toBe('[x]')
    expect(todoGlyph('in_progress')).toBe('[>]')
    expect(todoGlyph('pending')).toBe('[ ]')
    expect(todoGlyph('cancelled')).toBe('[-]')
  })
})

describe('todoTone', () => {
  it('keeps todo status rows neutral instead of red/green', () => {
    expect(todoTone('completed')).toBe('dim')
    expect(todoTone('cancelled')).toBe('dim')
    expect(todoTone('pending')).toBe('body')
    expect(todoTone('in_progress')).toBe('active')
  })
})

describe('todoTree', () => {
  it('orders parents before children with depths', () => {
    const tree = todoTree([
      { content: 'WP1', id: 'wp1', status: 'in_progress' },
      { content: 'WP2', id: 'wp2', status: 'pending' },
      { content: 'T1', id: 't1', parent: 'wp1', status: 'pending' },
      { content: 'T2', id: 't2', parent: 'wp1', status: 'pending' }
    ])

    expect(tree.map(([t, d]) => [t.id, d])).toEqual([
      ['wp1', 0],
      ['t1', 1],
      ['t2', 1],
      ['wp2', 0]
    ])
  })

  it('degrades dangling and self parents to roots', () => {
    const tree = todoTree([
      { content: 'A', id: 'a', parent: 'ghost', status: 'pending' },
      { content: 'B', id: 'b', parent: 'b', status: 'pending' }
    ])

    expect(tree.map(([t, d]) => [t.id, d])).toEqual([
      ['a', 0],
      ['b', 0]
    ])
  })

  it('keeps cycle members instead of dropping them', () => {
    const tree = todoTree([
      { content: 'A', id: 'a', parent: 'b', status: 'pending' },
      { content: 'B', id: 'b', parent: 'a', status: 'pending' }
    ])

    expect(tree.map(([t]) => t.id).sort()).toEqual(['a', 'b'])
  })

  it('flattens a todo list with no parents unchanged, all at depth 0', () => {
    const tree = todoTree([
      { content: 'A', id: 'a', status: 'pending' },
      { content: 'B', id: 'b', status: 'completed' }
    ])

    expect(tree.map(([t, d]) => [t.id, d])).toEqual([
      ['a', 0],
      ['b', 0]
    ])
  })
})
