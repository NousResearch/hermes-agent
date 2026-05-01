import { describe, expect, it } from 'vitest'

import { toTranscriptMessages } from '../domain/messages.js'
import { upsert } from '../lib/messages.js'

describe('toTranscriptMessages', () => {
  it('preserves assistant tool-call rows so resume does not drop prior turns', () => {
    const rows = [
      { role: 'user', text: 'first prompt' },
      { role: 'tool', context: 'repo', name: 'search_files', text: 'ignored raw result' },
      { role: 'assistant', text: 'first answer' },
      { role: 'user', text: 'second prompt' }
    ]

    expect(toTranscriptMessages(rows).map(msg => [msg.role, msg.text])).toEqual([
      ['user', 'first prompt'],
      ['assistant', 'first answer'],
      ['user', 'second prompt']
    ])
    expect(toTranscriptMessages(rows)[1]?.tools?.[0]).toContain('Search Files')
  })

  it('passes through thinking from assistant rows for resume', () => {
    const rows = [
      { role: 'user', text: 'question' },
      { role: 'assistant', text: 'answer', thinking: 'my reasoning' },
    ]

    const result = toTranscriptMessages(rows)
    expect(result).toHaveLength(2)
    expect(result[1]?.thinking).toBe('my reasoning')
  })

  it('does not include thinking when absent', () => {
    const rows = [
      { role: 'assistant', text: 'answer' },
    ]

    const result = toTranscriptMessages(rows)
    expect(result).toHaveLength(1)
    expect(result[0]?.thinking).toBeUndefined()
  })
})

describe('upsert', () => {
  it('appends when last role differs', () => {
    expect(upsert([{ role: 'user', text: 'hi' }], 'assistant', 'hello')).toHaveLength(2)
  })

  it('replaces when last role matches', () => {
    expect(upsert([{ role: 'assistant', text: 'partial' }], 'assistant', 'full')[0]!.text).toBe('full')
  })

  it('appends to empty', () => {
    expect(upsert([], 'user', 'first')).toEqual([{ role: 'user', text: 'first' }])
  })

  it('does not mutate', () => {
    const prev = [{ role: 'user' as const, text: 'hi' }]
    upsert(prev, 'assistant', 'yo')
    expect(prev).toHaveLength(1)
  })
})
