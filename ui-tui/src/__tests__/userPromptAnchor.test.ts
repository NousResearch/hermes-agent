import { describe, expect, it } from 'vitest'

import { userPromptAnchorStyle } from '../domain/userPromptAnchor.js'
import { DEFAULT_THEME } from '../theme.js'

describe('userPromptAnchorStyle', () => {
  it('uses a purple user-only anchor that does not reuse the active theme accent by default', () => {
    const style = userPromptAnchorStyle(DEFAULT_THEME)

    expect(style?.borderColor).toBe('#B084FF')
    expect(style?.titleColor).toBe('#B084FF')
    expect(style?.textColor).toBe('#F2EAFE')
    expect(style?.borderStyle).toBe('round')
    expect(style?.title).toBe('You')
    expect(style?.marginTop).toBe(2)
    expect(style?.marginBottom).toBe(2)

    expect(style?.borderColor).not.toBe(DEFAULT_THEME.color.primary)
    expect(style?.borderColor).not.toBe(DEFAULT_THEME.color.accent)
    expect(style?.textColor).not.toBe(DEFAULT_THEME.color.label)
  })

  it('honors global user_message_preview colors for blue-themed machines', () => {
    const style = userPromptAnchorStyle(DEFAULT_THEME, {
      accent_color: '#FFD700',
      margin_bottom: 2,
      margin_top: 2,
      text_color: '#FFF8DC'
    })

    expect(style?.borderColor).toBe('#FFD700')
    expect(style?.titleColor).toBe('#FFD700')
    expect(style?.textColor).toBe('#FFF8DC')
    expect(style?.marginTop).toBe(2)
    expect(style?.marginBottom).toBe(2)
  })

  it('can disable the TUI user anchor when boxed is false', () => {
    const style = userPromptAnchorStyle(DEFAULT_THEME, { boxed: false })

    expect(style).toBeNull()
  })
})
