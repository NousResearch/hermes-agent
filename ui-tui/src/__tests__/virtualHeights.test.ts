import { describe, expect, it } from 'vitest'

import { estimateBodyHeight, estimatedMsgHeight, messageHeightKey, wrappedLines } from '../lib/virtualHeights.js'
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

    expect(estimatedMsgHeight(msg, 30, { compact: false, details: false, userPrompt: '❯' })).toBe(3)
    expect(estimatedMsgHeight(msg, 30, { compact: false, details: false, userPrompt: 'Ψ >' })).toBe(4)
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

describe('fence-aware virtual heights', () => {
  // The renderer paints fenced code inside a panel whose inner width is
  // narrower than the body width (border + padding cost 4 cells in normal
  // mode, 2 cells in narrow/compact mode). The estimator must use the
  // same width or it will undercount and the virtual transcript spacer
  // will be too short, producing a visual jolt on first mount before
  // Yoga remeasures.
  it('wraps a long unbroken fenced line at the panel inner width, not the body width', () => {
    // Body width 30 → normal panel inner width 26. A 30-char source line
    // fits at body width (= 1 wrapped row) but wraps inside the panel to
    // 2 rows. The estimator must count 2 code rows + 2 chrome rows = 4
    // rows, NOT 1 + 2 = 3.
    const text = ['```python', 'x'.repeat(30), '```'].join('\n')
    const height = estimateBodyHeight(text, 30, false)

    expect(height).toBe(4) // 2 wrapped code rows + top + bottom border
  })

  it('matches the renderer at a body width near the 20-col threshold (normal vs narrow)', () => {
    // Body width 24 → normal panel, inner width 20. A 22-char source
    // line wraps to 2 rows inside the panel.
    const text24 = ['```python', 'x'.repeat(22), '```'].join('\n')
    expect(estimateBodyHeight(text24, 24, false)).toBe(4) // 2 code + 2 chrome

    // Body width 18 → narrow fallback, inner width 16. 22-char line
    // wraps to 2 code rows + 1 lang row = 3 rows.
    const text18 = ['```python', 'x'.repeat(22), '```'].join('\n')
    expect(estimateBodyHeight(text18, 18, false)).toBe(3) // 2 code + 1 lang
  })

  it('treats compact mode the same as narrow regardless of body width', () => {
    // Use a long enough source line that 22 cells won't fit even at the
    // narrow-mode inner width (bodyWidth 30 - narrow overhead 2 = 28).
    // 40 chars wraps to 2 rows.
    const text = ['```python', 'x'.repeat(40), '```'].join('\n')

    // 30-col body but `compact: true` → narrow layout → inner width 28.
    // 40 / 28 = 1.43 → ceil 2 wrapped rows + 1 lang row = 3.
    expect(estimateBodyHeight(text, 30, true)).toBe(3)
  })

  it('handles a no-language fence: chrome is 0 in narrow mode, 2 in normal mode', () => {
    const narrow = ['```', 'x = 1', '```'].join('\n')
    // narrow + no lang → 1 code row + 0 lang row = 1
    expect(estimateBodyHeight(narrow, 18, false)).toBe(1)

    const normal = ['```', 'x = 1', '```'].join('\n')
    // normal + no lang → 1 code row + top + bottom border = 3
    expect(estimateBodyHeight(normal, 24, false)).toBe(3)
  })

  it('mixes prose and fenced blocks in the same source', () => {
    // 2 lines of prose, then a 1-line code block, then 2 more lines.
    // 24-col body: prose wraps 1 row per line; code panel: 1 row + 2 chrome.
    const text = ['intro line', 'second line', '```python', 'x = 1', '```', 'after', 'final'].join('\n')
    const h = estimateBodyHeight(text, 24, false)
    // 2 prose rows + (1 code row + 2 chrome) + 2 prose rows = 7
    expect(h).toBe(7)
  })

  it('counts an empty fence as at least one wrapped code row plus the chrome', () => {
    // Empty body inside a fence: the renderer emits a single
    // `<Text> </Text>` line so the code section is at least 1 row.
    const normal = ['```python', '', '```'].join('\n')
    expect(estimateBodyHeight(normal, 24, false)).toBe(3) // 1 code + 2 chrome

    const narrow = ['```python', '', '```'].join('\n')
    expect(estimateBodyHeight(narrow, 18, false)).toBe(2) // 1 code + 1 lang
  })

  it('treats an unclosed fence as code all the way to end of text', () => {
    // Matches the renderer: if no closer is found, the rest of the
    // document is rendered as code. Estimator mirrors that.
    const text = ['```python', 'x = 1', 'still code'].join('\n')
    // normal panel: 2 code rows + 2 chrome = 4
    expect(estimateBodyHeight(text, 30, false)).toBe(4)
  })

  it('stays fast on a giant fenced line (no O(text) walk past the byte budget)', () => {
    // 1M-char code body inside a single fence. The estimator must not
    // walk the whole body — it should cap at the same MAX_ESTIMATE_LINES
    // bound the prose estimator uses.
    const text = ['```python', 'x'.repeat(1_000_000), '```'].join('\n')
    const t0 = performance.now()
    const h = estimateBodyHeight(text, 80, false)
    const elapsed = performance.now() - t0

    expect(h).toBeLessThanOrEqual(800)
    expect(elapsed).toBeLessThan(50)
  })

  it('estimatedMsgHeight on a long-fence assistant message does not undercount', () => {
    // Regression: a 30-char source line at body width 30 is 1 row at
    // body width but 2 rows inside the normal panel. The old estimator
    // counted 1 + 2 = 3; the new one must count at least 4.
    const msg: Msg = {
      role: 'assistant',
      text: ['```python', 'x'.repeat(30), '```'].join('\n')
    }

    const h = estimatedMsgHeight(msg, 35, { compact: false, details: false })

    // The body height must be at least 4 (2 code rows + 2 chrome rows).
    // The paragraph-gap step in `estimatedMsgHeight` only fires for source
    // blank lines, which this message has none of, so h is exactly the
    // panel height — no +1 gap row.
    expect(h).toBeGreaterThanOrEqual(4)
  })

  it('counts `md` / `markdown` fences as prose at the body width, not as a code panel', () => {
    // The renderer recurses `lang === 'md' || 'markdown'` fences through
    // `<Md cols={cols}>` instead of painting the rounded CodeBlock panel.
    // That means: body width is the full body width (no -4 panel
    // overhead), no top/bottom border rows, and the language label is
    // not rendered as a separate row. The estimator must mirror that or
    // it overcounts md/markdown fences by 2 (the missing panel chrome)
    // and by any inner-width-vs-body-width difference.
    //
    // Pre-fix the estimator returned 4 (= 2 wrapped code rows at
    // bodyWidth - 4 + 2 chrome) for the first case; the renderer
    // actually renders 2 rows of prose at the full body width.
    const longLine = 'a'.repeat(30)

    // `md` — 30 chars at bodyWidth 30 (innerW 26 would wrap to 2). At
    // bodyWidth 30 the 30-char line fits on 1 row. No chrome.
    const mdText = ['```md', longLine, '```'].join('\n')
    expect(estimateBodyHeight(mdText, 30, false)).toBe(1)

    // `markdown` — same expectation via the alternate lang string.
    const markdownText = ['```markdown', longLine, '```'].join('\n')
    expect(estimateBodyHeight(markdownText, 30, false)).toBe(1)

    // Even at the narrow threshold, no language row is added for a
    // markdown fence: the renderer paints it as a plain prose block.
    expect(estimateBodyHeight(mdText, 18, false)).toBe(2) // 30 chars / 18 ≈ 2 rows, no chrome

    // Sanity: a real code-language fence of the same shape still takes
    // the panel path. 30 chars / innerW 26 → 2 rows + 2 chrome = 4.
    expect(estimateBodyHeight(['```python', longLine, '```'].join('\n'), 30, false)).toBe(4)
  })
})
