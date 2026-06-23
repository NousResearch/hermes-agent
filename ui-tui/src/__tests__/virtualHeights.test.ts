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
    const msg: Msg = { role: 'user', text: 'x'.repeat(21) }

    expect(estimatedMsgHeight(msg, 26, { compact: false, details: false, userPrompt: '❯' })).toBe(3)
    expect(estimatedMsgHeight(msg, 26, { compact: false, details: false, userPrompt: 'Ψ >' })).toBe(4)
  })

  it('adds one row for a group-boundary lead gap', () => {
    const msg: Msg = { role: 'assistant', text: 'reply' }

    expect(estimatedMsgHeight(msg, 80, { compact: false, details: false, leadGap: true })).toBe(
      estimatedMsgHeight(msg, 80, { compact: false, details: false, leadGap: false }) + 1
    )
  })

  it('includes detail sections when visible', () => {
    const msg: Msg = { role: 'assistant', text: 'ok', thinking: 'line 1\nline 2', tools: ['Tool A', 'Tool B'] }

    expect(estimatedMsgHeight(msg, 80, { compact: false, details: true })).toBeGreaterThan(
      estimatedMsgHeight(msg, 80, { compact: false, details: false })
    )
  })

  it('accounts for the response separator when assistant details are visible', () => {
    const msg: Msg = { role: 'assistant', text: 'ok', thinking: 'plan' }

    expect(estimatedMsgHeight(msg, 80, { compact: false, details: true })).toBe(
      estimatedMsgHeight(msg, 80, { compact: false, details: false }) + 3
    )
  })

  it('does not account for a response separator without visible details', () => {
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

  it('treats historical thinking blocks as collapsed unless explicitly expanded', () => {
    const msg: Msg = { role: 'assistant', text: 'ok', thinking: 'line 1\nline 2\nline 3' }

    expect(
      estimatedMsgHeight(msg, 80, {
        compact: false,
        details: true,
        thinkingExpanded: false,
        thinkingVisible: true,
        toolsVisible: false
      })
    ).toBeLessThan(
      estimatedMsgHeight(msg, 80, {
        compact: false,
        details: true,
        thinkingExpanded: true,
        thinkingVisible: true,
        toolsVisible: false
      })
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
