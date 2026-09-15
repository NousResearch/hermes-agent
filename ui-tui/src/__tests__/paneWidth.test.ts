import { describe, expect, it } from 'vitest'

import { composerPromptWidth, stableComposerColumns, transcriptBodyWidth, transcriptGutterWidth, transcriptPaneCols } from '../lib/inputMetrics.js'

// Pane paddingX={1} (2 cols) + transcript scrollbar (1 col). Not an extra
// leftover reading-column reserve.
const PANE_CHROME = 3

describe('CW-01 pane-fill wrap', () => {
  it('assistant wrap uses extra columns when the pane widens (not a leftover reading column)', () => {
    const narrow = transcriptBodyWidth(80, 'assistant', '>')
    const wide = transcriptBodyWidth(160, 'assistant', '>')
    const gutter = transcriptGutterWidth('assistant', '>')

    expect(wide).toBeGreaterThan(narrow)
    expect(wide - narrow).toBe(80)
    expect(wide).toBe(160 - gutter - PANE_CHROME)
    expect(narrow).toBe(80 - gutter - PANE_CHROME)
  })

  it('narrow panes reflow without a 20-col floor that would clip into chrome', () => {
    const gutter = transcriptGutterWidth('assistant', '>')
    const wrap = transcriptBodyWidth(24, 'assistant', '>')

    expect(wrap).toBe(24 - gutter - PANE_CHROME)
    expect(wrap).toBeLessThan(20)
    expect(wrap + gutter + PANE_CHROME).toBe(24)
    expect(transcriptBodyWidth(10, 'user', '>', false)).toBeGreaterThanOrEqual(1)
  })

  it('composer wrap matches user transcript wrap after resize', () => {
    const prompt = '>'
    const promptWidth = composerPromptWidth(prompt)

    for (const cols of [20, 24, 40, 80, 100, 160]) {
      expect(stableComposerColumns(cols, promptWidth)).toBe(transcriptBodyWidth(cols, 'user', prompt))
    }

    const widePrompt = 'Ψ >'
    const widePromptWidth = composerPromptWidth(widePrompt)

    for (const cols of [40, 80, 120]) {
      expect(stableComposerColumns(cols, widePromptWidth)).toBe(transcriptBodyWidth(cols, 'user', widePrompt))
    }
  })

  it('does not steal rail or pet columns from the text pane', () => {
    expect(transcriptPaneCols(120, 0, null)).toBe(120)
    expect(transcriptPaneCols(120, 44, null)).toBe(76)
    expect(transcriptPaneCols(200, 0, 30)).toBe(170)
    expect(transcriptPaneCols(80, 0, 20)).toBe(80)
    expect(transcriptPaneCols(18, 0, null)).toBe(18)
  })
})
