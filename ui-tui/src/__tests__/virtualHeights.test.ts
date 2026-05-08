import { describe, expect, it } from 'vitest'

import { buildMessageCacheKey, estimatedMsgHeight, messageHeightKey, wrappedLines } from '../lib/virtualHeights.js'
import type { Msg } from '../types.js'

describe('virtual height estimates', () => {
  it('uses stable content keys across resumed message objects', () => {
    const msg: Msg = { role: 'assistant', text: 'same text', tools: ['Search Files [long message]'] }

    expect(messageHeightKey(msg)).toBe(messageHeightKey({ ...msg }))
  })

  it('accounts for wrapping and preserved blank-block rhythm', () => {
    const msg: Msg = { role: 'assistant', text: `one\n\n${'x'.repeat(90)}` }

    expect(wrappedLines(msg.text, 30)).toBe(5)
    expect(estimatedMsgHeight(msg, 35, { compact: false, details: false })).toBeGreaterThan(5)
  })

  it('uses compound user prompt width when estimating user message wrapping', () => {
    const msg: Msg = { role: 'user', text: 'x'.repeat(21) }

    expect(estimatedMsgHeight(msg, 26, { compact: false, details: false, userPrompt: '❯' })).toBe(3)
    expect(estimatedMsgHeight(msg, 26, { compact: false, details: false, userPrompt: 'Ψ >' })).toBe(4)
  })

  it('includes detail sections when visible', () => {
    const msg: Msg = { role: 'assistant', text: 'ok', thinking: 'line 1\nline 2', tools: ['Tool A', 'Tool B'] }

    expect(estimatedMsgHeight(msg, 80, { compact: false, details: true })).toBeGreaterThan(
      estimatedMsgHeight(msg, 80, { compact: false, details: false })
    )
  })

  it('counts wide unicode characters by display width, not code units', () => {
    // CJK characters are width 2 each
    const cjk = '这是一段中文测试文本'.repeat(4) // 40 chars, 80 display width
    const ascii = 'a'.repeat(80) // 80 chars, 80 display width

    // Both should wrap to 2 lines at width 40
    expect(wrappedLines(ascii, 40)).toBe(2)
    expect(wrappedLines(cjk, 40)).toBe(2)

    // At width 50, naive .length would give ceil(40/50)=1 for CJK,
    // but stringWidth correctly gives ceil(80/50)=2
    expect(wrappedLines(cjk, 50)).toBe(2)

    // At width 80, both fit in 1 line
    expect(wrappedLines(ascii, 80)).toBe(1)
    expect(wrappedLines(cjk, 80)).toBe(1)
  })

  it('counts CJK characters as double width', () => {
    const text = '这是一段中文测试文本'.repeat(4) // 40 CJK chars * 2 width = 80 display width

    expect(wrappedLines(text, 40)).toBe(2)
    expect(wrappedLines(text, 80)).toBe(1)
  })

  it('preserves empty lines as single rows even with wide chars', () => {
    expect(wrappedLines('hello\n\nworld', 40)).toBe(3)
    expect(wrappedLines('■\n\n■', 40)).toBe(3)
  })
})

describe('buildMessageCacheKey', () => {
  it('marks history items as truncated (H suffix)', () => {
    const msg: Msg = { role: 'assistant', text: 'hello' }
    const key = buildMessageCacheKey(msg, 0, 10, 3)

    expect(key.endsWith(':H')).toBe(true)
  })

  it('marks tail items as full-render (T suffix)', () => {
    const msg: Msg = { role: 'assistant', text: 'hello' }
    const key = buildMessageCacheKey(msg, 9, 10, 3)

    expect(key.endsWith(':T')).toBe(true)
  })

  it('changes key when a message slides from tail into history', () => {
    const msg: Msg = { role: 'assistant', text: 'hello' }
    const tailKey = buildMessageCacheKey(msg, 7, 10, 3)   // index 7 >= 7, tail
    const historyKey = buildMessageCacheKey(msg, 6, 10, 3) // index 6 < 7, history

    expect(tailKey.endsWith(':T')).toBe(true)
    expect(historyKey.endsWith(':H')).toBe(true)
    expect(tailKey).not.toBe(historyKey)
  })

  it('produces stable keys for identical messages at the same position', () => {
    const msg: Msg = { role: 'assistant', text: 'hello' }

    expect(buildMessageCacheKey(msg, 5, 10, 3)).toBe(buildMessageCacheKey({ ...msg }, 5, 10, 3))
  })
})
