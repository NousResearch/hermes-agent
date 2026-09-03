import { describe, expect, it } from 'vitest'

import type { AccountUsageInfo } from '../app/interfaces.js'
import { quotaModePreview } from '../components/quotaPicker.js'

const usage: AccountUsageInfo = {
  plan: 'plus',
  provider: 'openai-codex',
  windows: [
    { label: 'Session', remainingPercent: 97, resetAt: null, resetIn: '2h 13m', usedPercent: 3 },
    { label: 'Weekly', remainingPercent: 80, resetAt: null, resetIn: '4d 21h', usedPercent: 20 }
  ]
}

describe('quotaModePreview', () => {
  it('previews each mode exactly as the status bar renders it', () => {
    // One glyph per segment; the trailing window is appended bare.
    expect(quotaModePreview(usage, 'session')).toBe('◔ 97% 2h 13m')
    expect(quotaModePreview(usage, 'both')).toBe('◔ 97% 2h 13m · 80% 4d 21h')
    expect(quotaModePreview(usage, 'weekly')).toBe('◔ 80% 4d 21h')
    expect(quotaModePreview(usage, 'tightest')).toBe('◔ 80% 4d 21h')
  })

  it('says plainly that off renders nothing', () => {
    expect(quotaModePreview(usage, 'off')).toBe('(nothing)')
  })

  it('falls back to a dash when no snapshot has landed', () => {
    expect(quotaModePreview(null, 'session')).toBe('—')
    expect(quotaModePreview({ ...usage, windows: [] }, 'both')).toBe('—')
  })

  it('keeps previewing the off row without a snapshot', () => {
    expect(quotaModePreview(null, 'off')).toBe('(nothing)')
  })
})
