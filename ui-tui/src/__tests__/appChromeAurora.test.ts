import { describe, expect, it } from 'vitest'

import { buildAuroraStatusParts, formatPromptElapsed } from '../components/appChrome.js'
import { DARK_THEME, fromSkin } from '../theme.js'
import type { Usage } from '../types.js'

const usage: Usage = {
  calls: 1,
  context_max: 128_000,
  context_percent: 14,
  context_used: 18_000,
  input: 12_000,
  output: 6_000,
  total: 18_000
}

describe('Aurora TUI chrome', () => {
  it('builds the requested Aurora status sequence', () => {
    expect(
      buildAuroraStatusParts({
        model: 'gpt-5.5',
        promptElapsedMs: 8_000,
        sessionElapsedMs: 42 * 60_000,
        t: DARK_THEME,
        usage
      }).map(part => part.text)
    ).toEqual(['☤ Aurora Proto', ' │ gpt-5.5', ' │ ctx 18k/128k', ' │ [█░░░░░░░░░] 14%', ' │ 42m', ' │ ⏲ 8s'])
  })

  it('keeps the full Proto identity even when a skin shortens the agent name', () => {
    const skinned = fromSkin({}, { agent_name: 'Aurora' })

    expect(
      buildAuroraStatusParts({
        model: 'gpt-5.5',
        t: skinned,
        usage
      })[0]?.text
    ).toBe('☤ Aurora Proto')
  })

  it('uses the live timer glyph while a turn is running', () => {
    expect(formatPromptElapsed(12_500, true)).toBe('⏱ 12s')
    expect(formatPromptElapsed(12_500, false)).toBe('⏲ 12s')
  })
})
