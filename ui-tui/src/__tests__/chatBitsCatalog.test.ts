import { afterEach, describe, expect, it } from 'vitest'

import { applyLocale, messages, resetLocale, t } from '../i18n/runtime.js'

// The chat-bits sibling (branding, thinking, messageLine, todo, queued, entry)
// is registered in the en facade and resolves through the runtime — including
// the non-React `messages()` path entry.tsx uses at call time, which must
// observe a locale installed after import.
describe('chatBits catalog', () => {
  afterEach(() => resetLocale())

  it('is registered under the chatBits namespace with function leaves intact', () => {
    expect(messages().chatBits.thinking.toolCalls).toBe('Tool calls')
    expect(messages().chatBits.queued.header(3)).toBe('queued (3)')
    expect(t('chatBits.messageLine.chars', '1,234')).toBe('1,234 chars')
  })

  it('observes a locale swap through messages() and t()', () => {
    applyLocale('xx', {
      lang: 'xx',
      surface: 'tui',
      messages: { 'chatBits.entry.noTty': 'kein TTY', 'chatBits.branding.mcpSummary': '{0} MCPs' }
    })

    expect(messages().chatBits.entry.noTty).toBe('kein TTY')
    expect(messages().chatBits.branding.mcpSummary(2)).toBe('2 MCPs')
    expect(t('chatBits.thinking.thinking')).toBe('Thinking')

    resetLocale()

    expect(messages().chatBits.entry.noTty).toBe('hermes-tui: no TTY')
  })
})
