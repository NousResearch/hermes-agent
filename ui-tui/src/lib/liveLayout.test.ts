import { describe, expect, it } from 'vitest'

import { liveTailOrder } from './liveLayout.js'

describe('liveTailOrder', () => {
  it('keeps todo before transcript and assistant live output', () => {
    expect(liveTailOrder()).toEqual(['todo', 'history', 'assistant'])
  })
})
