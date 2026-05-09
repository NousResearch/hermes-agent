import { describe, expect, it } from 'vitest'

import {
  cleanThinkingText,
  flat,
  formatToolCall,
  hasAnsi,
  isPasteBackedText,
  isTransientTrailLine,
  parseToolTrailResultLine,
  splitToolDuration,
  stripAnsi,
  stripTrailingPasteNewlines,
  toolTrailLabel
} from '../lib/text.js'

const ESC = String.fromCharCode(27)

describe('hasAnsi', () => {
  it('detects CSI sequences', () => {
    expect(hasAnsi(`${ESC}[31mred${ESC}[0m`)).toBe(true)
  })

  it('detects OSC sequences', () => {
    expect(hasAnsi(`${ESC}]0;title${ESC}\\`)).toBe(true)
  })

  it('returns false for plain text', () => {
    expect(hasAnsi('plain text without escapes')).toBe(false)
  })

  it('returns false for empty string', () => {
    expect(hasAnsi('')).toBe(false)
  })
})

describe('stripAnsi', () => {
  it('removes CSI color sequences', () => {
    expect(stripAnsi(`${ESC}[31mred${ESC}[0m`)).toBe('red')
  })

  it('leaves text without escapes untouched', () => {
    expect(stripAnsi('hello world')).toBe('hello world')
  })
})

describe('stripTrailingPasteNewlines', () => {
  it('removes trailing newlines from non-empty text', () => {
    expect(stripTrailingPasteNewlines('hello\n\n\n')).toBe('hello')
  })

  it('preserves intermediate newlines', () => {
    expect(stripTrailingPasteNewlines('a\nb\n')).toBe('a\nb')
  })

  it('returns text unchanged when only newlines (whitespace fallback)', () => {
    expect(stripTrailingPasteNewlines('\n\n\n')).toBe('\n\n\n')
  })

  it('returns text unchanged when no trailing newline', () => {
    expect(stripTrailingPasteNewlines('hello')).toBe('hello')
  })
})

describe('toolTrailLabel', () => {
  it('title-cases snake_case', () => {
    expect(toolTrailLabel('read_file')).toBe('Read File')
  })

  it('handles single word', () => {
    expect(toolTrailLabel('bash')).toBe('Bash')
  })

  it('drops empty parts from leading underscore', () => {
    expect(toolTrailLabel('_get_status')).toBe('Get Status')
  })

  it('falls back to original for empty input', () => {
    expect(toolTrailLabel('')).toBe('')
  })
})

describe('formatToolCall', () => {
  it('appends compact preview in quotes when context present', () => {
    expect(formatToolCall('read_file', 'src/app.ts')).toBe('Read File("src/app.ts")')
  })

  it('returns label only when context empty', () => {
    expect(formatToolCall('bash')).toBe('Bash')
  })

  it('truncates long context to 64 chars', () => {
    const ctx = 'a'.repeat(100)
    const out = formatToolCall('write', ctx)
    expect(out.startsWith('Write("')).toBe(true)
    expect(out.endsWith('…")')).toBe(true)
  })

  it('collapses whitespace in context', () => {
    expect(formatToolCall('grep', '  foo\n  bar  ')).toBe('Grep("foo bar")')
  })
})

describe('parseToolTrailResultLine', () => {
  it('parses success mark with detail via " :: "', () => {
    expect(parseToolTrailResultLine('Read("a") :: 12 lines ✓')).toEqual({
      call: 'Read("a")',
      detail: '12 lines',
      mark: '✓'
    })
  })

  it('parses error mark', () => {
    expect(parseToolTrailResultLine('Bash("x") :: failed ✗')).toEqual({
      call: 'Bash("x")',
      detail: 'failed',
      mark: '✗'
    })
  })

  it('falls back to legacy ": " separator', () => {
    expect(parseToolTrailResultLine('Read: ok ✓')).toEqual({ call: 'Read', detail: 'ok', mark: '✓' })
  })

  it('returns body as call when no separator', () => {
    expect(parseToolTrailResultLine('Bash ✓')).toEqual({ call: 'Bash', detail: '', mark: '✓' })
  })

  it('returns null when not a result line', () => {
    expect(parseToolTrailResultLine('drafting answer')).toBeNull()
  })
})

describe('splitToolDuration', () => {
  it('extracts trailing duration', () => {
    expect(splitToolDuration('Read("x") (1.5s)')).toEqual({ label: 'Read("x")', duration: ' (1.5s)' })
  })

  it('handles integer seconds', () => {
    expect(splitToolDuration('Bash (12s)')).toEqual({ label: 'Bash', duration: ' (12s)' })
  })

  it('returns empty duration when none present', () => {
    expect(splitToolDuration('Read("x")')).toEqual({ label: 'Read("x")', duration: '' })
  })
})

describe('isTransientTrailLine', () => {
  it('matches drafting prefix', () => {
    expect(isTransientTrailLine('drafting reply…')).toBe(true)
  })

  it('matches exact "analyzing tool output…"', () => {
    expect(isTransientTrailLine('analyzing tool output…')).toBe(true)
  })

  it('rejects unrelated lines', () => {
    expect(isTransientTrailLine('Bash ✓')).toBe(false)
  })
})

describe('isPasteBackedText', () => {
  it('detects [[paste:N]] token', () => {
    expect(isPasteBackedText('hello [[paste:42]] world')).toBe(true)
  })

  it('detects "[paste #N attached]" form', () => {
    expect(isPasteBackedText('see [paste #1 attached]')).toBe(true)
  })

  it('detects "[paste #N excerpt]" form', () => {
    expect(isPasteBackedText('see [paste #2 excerpt: foo]')).toBe(true)
  })

  it('returns false when no paste markers', () => {
    expect(isPasteBackedText('plain text without paste tokens')).toBe(false)
  })
})

describe('flat', () => {
  it('flattens record values into one array', () => {
    expect(flat({ a: ['x', 'y'], b: ['z'] })).toEqual(['x', 'y', 'z'])
  })

  it('returns empty array for empty record', () => {
    expect(flat({})).toEqual([])
  })

  it('preserves order of insertion across keys', () => {
    expect(flat({ first: ['1'], second: ['2', '3'] })).toEqual(['1', '2', '3'])
  })
})

describe('cleanThinkingText', () => {
  it('returns trimmed input when no status verbs to strip', () => {
    expect(cleanThinkingText('Reasoning about edge case.')).toBe('Reasoning about edge case.')
  })

  it('drops blank lines between content', () => {
    expect(cleanThinkingText('a\n\n\n\nb')).toBe('a\nb')
  })

  it('inserts blank line before bold markers', () => {
    expect(cleanThinkingText('intro **bold**')).toBe('intro \n\n**bold**')
  })

  it('returns empty string when input is whitespace only', () => {
    expect(cleanThinkingText('   \n\n   ')).toBe('')
  })
})
