import { describe, expect, it } from 'vitest'

import { composerDividerLine } from '../components/appChrome.js'

describe('composer divider line', () => {
  it('spans the padded composer width', () => {
    expect(composerDividerLine(80)).toBe('─'.repeat(78))
  })

  it('keeps a visible rule on narrow panes', () => {
    expect(composerDividerLine(0)).toBe('─')
    expect(composerDividerLine(2)).toBe('─')
  })

  it('floors fractional terminal widths before subtracting padding', () => {
    expect(composerDividerLine(12.9)).toBe('─'.repeat(10))
  })
})
