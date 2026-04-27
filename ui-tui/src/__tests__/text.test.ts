import { describe, expect, it } from 'vitest'

import {
  boundedHistoryRenderText,
  boundedLiveRenderText,
  buildToolTrailLine,
  edgePreview,
  estimateRows,
  estimateTokensRough,
  fmtK,
  isToolTrailResultLine,
  lastCotTrailIndex,
  parseToolTrailResultLine,
  pasteTokenLabel,
  sameToolTrailGroup,
  splitToolDuration,
  thinkingPreview
} from '../lib/text.js'

describe('isToolTrailResultLine', () => {
  it('detects completion markers', () => {
    expect(isToolTrailResultLine('foo ✓')).toBe(true)
    expect(isToolTrailResultLine('foo ✗')).toBe(true)
    expect(isToolTrailResultLine('drafting x…')).toBe(false)
  })
})

describe('buildToolTrailLine', () => {
  it('puts completion duration inline before the result marker', () => {
    const line = buildToolTrailLine('read_file', 'x', false, '', 0.94)

    expect(line).toBe('Read File("x") (0.9s) ✓')
    expect(parseToolTrailResultLine(line)).toEqual({ call: 'Read File("x") (0.9s)', detail: '', mark: '✓' })
    expect(splitToolDuration('Read File("x") (0.9s)')).toEqual({ label: 'Read File("x")', duration: ' (0.9s)' })
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

  it('formats thousands as lowercase k', () => {
    expect(fmtK(1000)).toBe('1k')
    expect(fmtK(1500)).toBe('1.5k')
  })

  it('formats millions and billions with lowercase suffixes', () => {
    expect(fmtK(1_000_000)).toBe('1m')
    expect(fmtK(1_000_000_000)).toBe('1b')
  })
})

describe('estimateTokensRough', () => {
  it('uses 4 chars per token rounding up', () => {
    expect(estimateTokensRough('')).toBe(0)
    expect(estimateTokensRough('a')).toBe(1)
    expect(estimateTokensRough('abcd')).toBe(1)
    expect(estimateTokensRough('abcde')).toBe(2)
  })
})

describe('thinkingPreview', () => {
  it('adds paragraph breaks before markdown thinking headings', () => {
    const raw =
      '**Considering user instructions**\nI need to answer.**Planning tool execution**\nI can run tools.**Determining weather search parameters**\nUse SF.'

    expect(thinkingPreview(raw, 'full')).toBe(
      '**Considering user instructions**\nI need to answer.\n\n**Planning tool execution**\nI can run tools.\n\n**Determining weather search parameters**\nUse SF.'
    )
  })
})

describe('boundedLiveRenderText', () => {
  it('preserves short live text verbatim', () => {
    expect(boundedLiveRenderText('one\ntwo', { maxChars: 100, maxLines: 10 })).toBe('one\ntwo')
  })

  it('keeps the live tail by character budget', () => {
    const out = boundedLiveRenderText('abcdefghij', { maxChars: 4, maxLines: 10 })

    expect(out).toContain('ghij')
    expect(out).toContain('omitted')
    expect(out).not.toContain('abcdef')
  })

  it('keeps the live tail by line budget', () => {
    const out = boundedLiveRenderText(['a', 'b', 'c', 'd'].join('\n'), { maxChars: 100, maxLines: 2 })

    expect(out).toContain('c\nd')
    expect(out).toContain('omitted 2 lines')
    expect(out).not.toContain('a\nb')
  })
})

describe('boundedHistoryRenderText', () => {
  it('uses a non-live omission label for completed history', () => {
    const out = boundedHistoryRenderText('abcdefghij', { maxChars: 4, maxLines: 10 })

    expect(out).toContain('[showing tail; omitted')
    expect(out).not.toContain('live tail')
  })
})

describe('edgePreview', () => {
  it('keeps both ends for long text', () => {
    expect(edgePreview('Vampire Bondage ropes slipped from her neck, still stained with blood', 8, 18)).toBe(
      'Vampire.. stained with blood'
    )
  })
})

describe('pasteTokenLabel', () => {
  it('builds readable long-paste labels with counts', () => {
    const label = pasteTokenLabel('Vampire Bondage ropes slipped from her neck, still stained with blood', 250)
    expect(label.startsWith('[[ ')).toBe(true)
    expect(label).toContain('[250 lines]')
    expect(label.endsWith(' ]]')).toBe(true)
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

  it('keeps intraword underscores when sizing snake_case identifiers', () => {
    const w = 80
    const snake = 'look at test_case_with_underscores now'
    const plain = 'look at test case with underscores now'

    expect(estimateRows(snake, w)).toBe(estimateRows(plain, w))
  })
})
