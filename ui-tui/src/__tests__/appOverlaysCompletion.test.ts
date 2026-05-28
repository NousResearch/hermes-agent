import { describe, expect, it } from 'vitest'

import { completionOverlayMode, completionRowStyle } from '../components/appOverlays.js'
import { DEFAULT_THEME } from '../theme.js'

describe('completionOverlayMode', () => {
  it('renders slash completions in normal layout flow instead of absolute-overlapping the composer', () => {
    expect(completionOverlayMode()).toBe('flow')
  })
})

describe('completionRowStyle', () => {
  it('keeps inactive slash completions free of row and meta backgrounds', () => {
    const style = completionRowStyle(false, DEFAULT_THEME)

    expect(style.rowBackground).toBeUndefined()
    expect(style.commandColor).toBe(DEFAULT_THEME.color.label)
    expect(style.metaColor).toBe(DEFAULT_THEME.color.muted)
  })

  it('uses a dark status background and brighter text only for the active row', () => {
    const style = completionRowStyle(true, DEFAULT_THEME)

    expect(style.rowBackground).toBe(DEFAULT_THEME.color.statusBg)
    expect(style.commandColor).toBe(DEFAULT_THEME.color.primary)
    expect(style.metaColor).toBe(DEFAULT_THEME.color.text)
  })
})
