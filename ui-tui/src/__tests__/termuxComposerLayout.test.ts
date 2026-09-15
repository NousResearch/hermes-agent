import { describe, expect, it } from 'vitest'

import { stableComposerColumns, transcriptBodyWidth } from '../lib/inputMetrics.js'
import { composerPromptText } from '../lib/prompt.js'

describe('Termux composer prompt + width guards', () => {
  it('uses a single-cell ASCII prompt marker in Termux mode', () => {
    expect(composerPromptText('❯', 'coder', false, true, 50)).toBe('>')
  })

  it('suppresses profile prefixes on narrow Termux panes', () => {
    expect(composerPromptText('❯', 'upstr', false, true, 72)).toBe('>')
  })

  it('keeps profile context on very wide Termux panes', () => {
    expect(composerPromptText('❯', 'upstr', false, true, 120)).toBe('upstr >')
  })

  it('reserves padding only on Termux; desktop also reserves the scrollbar cell', () => {
    // 32 columns after prompt: both modes keep paddingX (2). Desktop also
    // shares the transcript scrollbar cell so composer and history match.
    expect(stableComposerColumns(40, 8, false)).toBe(29)
    expect(stableComposerColumns(40, 8, true)).toBe(30)

    // With ample room, Termux still skips the scrollbar cell.
    expect(stableComposerColumns(60, 8, true)).toBe(50)
  })

  it('never over-allocates transcript body width on narrow panes', () => {
    // Old behavior hard-minned to 20 columns and overflowed narrow layouts.
    expect(transcriptBodyWidth(24, 'assistant', '>', true)).toBe(19)
    expect(transcriptBodyWidth(24, 'user', 'upstr >', true)).toBe(14)
    expect(transcriptBodyWidth(10, 'user', '>', true)).toBeGreaterThanOrEqual(1)
  })

  it('does not floor desktop wrap at 20 on narrow panes', () => {
    expect(transcriptBodyWidth(24, 'assistant', '>')).toBe(18)
    expect(transcriptBodyWidth(24, 'user', 'upstr >')).toBe(13)
  })
})
