import { describe, expect, it } from 'vitest'

import { todoGlyph } from './todo.js'

describe('todoGlyph', () => {
  it('uses fixed-width ASCII markers so the active row does not render wide or emoji-like', () => {
    expect(todoGlyph('completed')).toBe('[x]')
    expect(todoGlyph('in_progress')).toBe('[>]')
    expect(todoGlyph('pending')).toBe('[ ]')
    expect(todoGlyph('cancelled')).toBe('[-]')
  })
})
