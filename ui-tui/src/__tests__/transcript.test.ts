import { describe, expect, it } from 'vitest'

import {
  exportTranscriptJson,
  formatTranscript,
  searchTranscript,
  visibleConversationItems
} from '../domain/transcript.js'
import type { Msg } from '../types.js'

const messages: Msg[] = [
  { role: 'system', text: 'booting' },
  { role: 'user', text: 'hello world' },
  { role: 'assistant', text: '' },
  { role: 'tool', text: 'tool output mentions world' },
  { role: 'assistant', text: 'second answer with Needle\nnext line' }
]

describe('transcript helpers', () => {
  it('keeps visible conversation roles and drops system messages', () => {
    expect(visibleConversationItems(messages).map(m => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
  })

  it('formats transcript messages with stable role labels and fallback bodies', () => {
    const output = formatTranscript(messages, { previewChars: 12 })

    expect(output).toContain('[You #1]\nhello world')
    expect(output).toContain('[Hermes #2]\n(empty)')
    expect(output).toContain('[Tool #3]\ntool output…')
    expect(output).toContain('[Hermes #4]\nsecond answe…')
    expect(output).not.toContain('booting')
  })

  it('searches transcript text case-insensitively and returns matching excerpts', () => {
    const result = searchTranscript(messages, 'needle', { previewChars: 30 })

    expect(result.count).toBe(1)
    expect(result.text).toContain('[Hermes #4]')
    expect(result.text).toContain('Needle')
  })

  it('returns an empty search result for blank queries', () => {
    expect(searchTranscript(messages, '   ').count).toBe(0)
  })

  it('exports visible transcript messages to stable JSON', () => {
    const json = exportTranscriptJson(messages, { sessionId: 'sid-1', title: 'Demo' })
    const parsed = JSON.parse(json)

    expect(parsed.session_id).toBe('sid-1')
    expect(parsed.title).toBe('Demo')
    expect(parsed.messages).toHaveLength(4)
    expect(parsed.messages[0]).toEqual({ role: 'user', text: 'hello world' })
    expect(json.endsWith('\n')).toBe(true)
  })
})
