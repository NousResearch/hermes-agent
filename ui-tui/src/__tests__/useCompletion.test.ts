import { describe, expect, it } from 'vitest'

import { completionRequestForInput } from '../hooks/useCompletion.js'

describe('completionRequestForInput', () => {
  it('routes real slash commands to slash completion', () => {
    expect(completionRequestForInput('/help')).toMatchObject({
      method: 'complete.slash',
      params: { text: '/help' },
      replaceFrom: 1
    })
  })

  it('routes trailing context references in skill commands to path completion', () => {
    const input = '/plan Read requirements from @file'

    expect(completionRequestForInput(input)).toMatchObject({
      method: 'complete.path',
      params: { word: '@file' },
      replaceFrom: input.indexOf('@file')
    })
  })

  it('opens context completion after typing bare at in a skill command', () => {
    const input = '/plan Read requirements from @'

    expect(completionRequestForInput(input)).toMatchObject({
      method: 'complete.path',
      params: { word: '@' },
      replaceFrom: input.indexOf('@')
    })
  })

  it('keeps slash completion for non-context command arguments', () => {
    const input = '/plan Read requirements from ./docs'

    expect(completionRequestForInput(input)).toMatchObject({
      method: 'complete.slash',
      params: { text: input },
      replaceFrom: 1
    })
  })

  it('does not route absolute paths through slash completion', () => {
    expect(
      completionRequestForInput('/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md')
    ).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md' },
      replaceFrom: 0
    })
  })

  it('keeps path completion for trailing absolute path tokens', () => {
    expect(completionRequestForInput('read /home/d/Desktop/file.md')).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/file.md' },
      replaceFrom: 5
    })
  })

  it('leaves plain text alone', () => {
    expect(completionRequestForInput('hello there')).toBeNull()
  })
})
