import { describe, expect, it } from 'vitest'

import { terminalTitleText } from './use-terminal-title.js'

describe('terminalTitleText', () => {
  it('keeps every title byte printable ASCII inside GNU Screen', () => {
    const env = { STY: '1234.pts-0.host' } as NodeJS.ProcessEnv

    expect(terminalTitleText('✓ résumé · gpt-5 · ~/x', env)).toBe('+ r?sum? - gpt-5 - ~/x')
    expect(terminalTitleText('⏳ busy', env)).toBe('* busy')
    expect(terminalTitleText('⚠ \u001b[1mwait\u001b[0m', env)).toBe('! wait')
  })

  it('leaves Unicode titles alone outside Screen', () => {
    expect(terminalTitleText('✓ résumé · gpt-5', {} as NodeJS.ProcessEnv)).toBe('✓ résumé · gpt-5')
  })
})
