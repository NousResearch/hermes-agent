import { describe, expect, it } from 'vitest'

import { INLINE_IMAGE_MAX_ROWS } from '../config/limits.js'
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
    // cols must clear the 20-col body-width floor for both prompts (gutter +
    // horizontalReserve=4) so the wider 'Ψ >' prompt actually narrows the
    // body enough to wrap an extra line vs the single-cell '❯' prompt.
    const msg: Msg = { role: 'user', text: 'x'.repeat(23) }

    // 1-2 wrapped body rows + 2 margin rows + 2 shade-bar padding rows.
    expect(estimatedMsgHeight(msg, 30, { compact: false, details: false, userPrompt: '❯' })).toBe(5)
    expect(estimatedMsgHeight(msg, 30, { compact: false, details: false, userPrompt: 'Ψ >' })).toBe(6)
  })

  it('reserves the shade-bar padding rows for user prompts only', () => {
    const prompt: Msg = { role: 'user', text: 'hi' }
    const slash: Msg = { kind: 'slash', role: 'user', text: '/help' }
    const diff: Msg = { kind: 'diff', role: 'assistant', text: '+a' }

    expect(estimatedMsgHeight(prompt, 80, { compact: false, details: false })).toBe(1 + 2 + 2)
    expect(estimatedMsgHeight(diff, 80, { compact: false, details: false })).toBe(1 + 2)
    expect(estimatedMsgHeight(slash, 80, { compact: false, details: false })).toBeLessThan(1 + 2 + 2)
  })

  it('reserves a tool image at its cap, with no text line or gap, and attachments under a gap', () => {
    const tool: Msg = { images: ['/tmp/a.png'], kind: 'image', role: 'system', text: '' }
    const user: Msg = { images: ['/tmp/a.png'], role: 'user', text: 'what is this?' }
    const plainUser: Msg = { role: 'user', text: 'what is this?' }

    expect(estimatedMsgHeight(tool, 100, { compact: false, details: true })).toBe(INLINE_IMAGE_MAX_ROWS)
    expect(
      estimatedMsgHeight(user, 100, { compact: false, details: true }) -
        estimatedMsgHeight(plainUser, 100, { compact: false, details: true })
    ).toBe(1 + INLINE_IMAGE_MAX_ROWS)
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
    const rows = wrappedLines(giant, 80)

    expect(rows).toBeLessThanOrEqual(800)
  })
})
