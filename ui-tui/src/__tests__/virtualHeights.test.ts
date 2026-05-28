import { describe, expect, it } from 'vitest'

import { estimatedMsgHeight, messageHeightKey, wrappedLines } from '../lib/virtualHeights.js'
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
    const msg: Msg = { role: 'user', text: 'x'.repeat(23) }

    expect(estimatedMsgHeight(msg, 30, { compact: false, details: false, userPrompt: '❯' })).toBe(3)
    expect(estimatedMsgHeight(msg, 30, { compact: false, details: false, userPrompt: 'Ψ >' })).toBe(4)
  })

  it('includes detail sections when visible', () => {
    const msg: Msg = { role: 'assistant', text: 'ok', thinking: 'line 1\nline 2', tools: ['Tool A', 'Tool B'] }

    expect(estimatedMsgHeight(msg, 80, { compact: false, details: true })).toBeGreaterThan(
      estimatedMsgHeight(msg, 80, { compact: false, details: false })
    )
  })

  it('accounts for attached action-feed details when assistant details are visible', () => {
    const msg: Msg = { role: 'assistant', text: 'ok', thinking: 'plan' }

    expect(estimatedMsgHeight(msg, 80, { compact: false, details: true })).toBe(
      estimatedMsgHeight(msg, 80, { compact: false, details: false }) + 2
    )
  })

  it('does not account for attached details without visible details', () => {
    const msg: Msg = { role: 'assistant', text: 'ok' }

    expect(estimatedMsgHeight(msg, 80, { compact: false, details: true })).toBe(
      estimatedMsgHeight(msg, 80, { compact: false, details: false })
    )
  })

  it('honors per-section visibility when estimating response separators', () => {
    const thinkingOnly: Msg = { role: 'assistant', text: 'ok', thinking: 'plan' }
    const toolsOnly: Msg = { role: 'assistant', text: 'ok', tools: ['Tool A'] }

    expect(
      estimatedMsgHeight(thinkingOnly, 80, {
        compact: false,
        details: true,
        thinkingVisible: false,
        toolsVisible: true
      })
    ).toBe(estimatedMsgHeight(thinkingOnly, 80, { compact: false, details: false }))

    expect(
      estimatedMsgHeight(toolsOnly, 80, {
        compact: false,
        details: true,
        thinkingVisible: true,
        toolsVisible: false
      })
    ).toBe(estimatedMsgHeight(toolsOnly, 80, { compact: false, details: false }))
  })

  it('accounts for assistant log prefix rows', () => {
    const plain: Msg = { role: 'assistant', text: '파일 확인 중입니다.' }
    const prefixed: Msg = { role: 'assistant', text: '진행상황: 파일 확인 중입니다.' }

    expect(estimatedMsgHeight(prefixed, 80, { compact: false, details: false })).toBe(
      estimatedMsgHeight(plain, 80, { compact: false, details: false }) + 1
    )
  })

  it('reserves the action-feed bottom gap for visible trail rows', () => {
    const trail: Msg = { kind: 'trail', role: 'system', text: '', tools: ['Terminal("npm test")'] }

    expect(estimatedMsgHeight(trail, 80, { compact: false, details: true })).toBe(
      estimatedMsgHeight(trail, 80, { compact: false, details: true, toolsVisible: false }) + 2
    )
  })

  it('reserves two extra rows for the inter-turn separator on non-first user messages', () => {
    const msg: Msg = { role: 'user', text: 'follow-up question' }
    const base = estimatedMsgHeight(msg, 80, { compact: false, details: false })
    const withSep = estimatedMsgHeight(msg, 80, { compact: false, details: false, withSeparator: true })

    expect(withSep).toBe(base + 2)
  })

  it('caps wrapped-line counting so giant assistant turns do not block offset rebuilds', () => {
    // wrappedLines is invoked once per uncached message during
    // useVirtualHistory's offset rebuild. Unbounded counting on a long
    // assistant response (10k+ chars × every row × every rebuild) blocks
    // the UI on cold mount. Cap is ~800 rows; post-mount Yoga
    // measurement converges to the true height regardless.
    const giant = 'x'.repeat(1_000_000)
    const t0 = performance.now()
    const rows = wrappedLines(giant, 80)
    const elapsed = performance.now() - t0

    expect(rows).toBeLessThanOrEqual(800)
    expect(elapsed).toBeLessThan(50)
  })
})
