import { describe, expect, it } from 'vitest'

import { modelLabel } from '../components/appChrome.js'

const VS16 = '\uFE0F'

describe('modelLabel', () => {
  it('shows a lightning bolt in fast mode', () => {
    expect(modelLabel('demo-model', 'high', true).startsWith(`⚡${VS16} `)).toBe(true)
    expect(modelLabel('demo-model', 'high', true)).toContain(' fast')
  })

  it('shows a turtle when not in fast mode', () => {
    expect(modelLabel('demo-model', 'high', false).startsWith('🐢 ')).toBe(true)
    expect(modelLabel('demo-model', 'high', false)).not.toContain(' fast')
  })
})
