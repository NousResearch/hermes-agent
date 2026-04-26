import { describe, expect, it } from 'vitest'

import { liveTailOrder } from './liveLayout.js'

describe('liveTailOrder', () => {
  it('anchors live todo after scroll history and assistant output', () => {
    expect(liveTailOrder()).toEqual(['scroll-history', 'assistant', 'live-todo'])
  })
})
