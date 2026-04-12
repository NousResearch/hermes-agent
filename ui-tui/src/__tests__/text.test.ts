import { describe, expect, it } from 'vitest'

import { estimateRows, fmtK, isToolTrailResultLine, lastCotTrailIndex, sameToolTrailGroup } from '../lib/text.js'

describe('isToolTrailResultLine', () => {
  it('detects completion markers', () => {
    expect(isToolTrailResultLine('foo ✓')).toBe(true)
    expect(isToolTrailResultLine('foo ✗')).toBe(true)
    expect(isToolTrailResultLine('drafting x…')).toBe(false)
  })
})

describe('lastCotTrailIndex', () => {
  it('finds last non-result line', () => {
    expect(lastCotTrailIndex(['a ✓', 'thinking…'])).toBe(1)
    expect(lastCotTrailIndex(['only result ✓'])).toBe(-1)
  })
})

describe('sameToolTrailGroup', () => {
  it('matches bare check lines', () => {
    expect(sameToolTrailGroup('searching', 'searching ✓')).toBe(true)
    expect(sameToolTrailGroup('searching', 'searching ✗')).toBe(true)
  })

  it('matches contextual lines', () => {
    expect(sameToolTrailGroup('searching', 'searching: * ✓')).toBe(true)
    expect(sameToolTrailGroup('searching', 'searching: foo ✓')).toBe(true)
  })

  it('rejects other tools', () => {
    expect(sameToolTrailGroup('searching', 'reading ✓')).toBe(false)
    expect(sameToolTrailGroup('searching', 'searching extra ✓')).toBe(false)
  })
})

describe('fmtK', () => {
  it('keeps small numbers plain', () => {
    expect(fmtK(999)).toBe('999')
  })

  it('formats thousands as K', () => {
    expect(fmtK(1000)).toBe('1K')
    expect(fmtK(1500)).toBe('1.5K')
  })

  it('formats millions and billions', () => {
    expect(fmtK(1_000_000)).toBe('1M')
    expect(fmtK(1_000_000_000)).toBe('1B')
  })
})

describe('estimateRows', () => {
  it('handles tilde code fences', () => {
    const md = ['~~~markdown', '# heading', '~~~'].join('\n')

    expect(estimateRows(md, 40)).toBeGreaterThanOrEqual(2)
  })

  it('handles checklist bullets as list rows', () => {
    const md = ['- [x] done', '- [ ] todo'].join('\n')

    expect(estimateRows(md, 40)).toBe(2)
  })
})
