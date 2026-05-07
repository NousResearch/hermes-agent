import { describe, expect, it } from 'vitest'

import {
  breakTerminalShapingRuns,
  statusTextAnsi,
  stripStatusAnsi,
  truncateStatusText
} from '../components/shapeSafeStatusText.js'

describe('shape-safe status text', () => {
  it('keeps the Gemini model label visibly unchanged while breaking the terminal shaping run', () => {
    const label = 'gemini 3.1 pro preview high'
    const ansi = statusTextAnsi(label, '#cc9b1f')

    expect(stripStatusAnsi(ansi)).toBe(label)
    expect(ansi).toContain('gemin\x1b[25mi')
  })

  it('only inserts the shaping boundary for standalone Gemini labels', () => {
    expect(breakTerminalShapingRuns('minimax/minimax-m2.5')).toBe('minimax/minimax-m2.5')
    expect(breakTerminalShapingRuns('gemini 3.1')).toBe('gemin\x1b[25mi 3.1')
    expect(breakTerminalShapingRuns('Gemini 3.1')).toBe('Gemin\x1b[25mi 3.1')
  })

  it('truncates by terminal display width before adding ANSI', () => {
    expect(truncateStatusText('gemini 3.1 pro', 10)).toBe('gemini 3.…')
    expect(stripStatusAnsi(statusTextAnsi(truncateStatusText('gemini 3.1 pro', 10), '#cc9b1f'))).toBe('gemini 3.…')
  })
})
