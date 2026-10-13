import { describe, expect, it } from 'vitest'

import { completionParams, completionRequestForInput } from '../hooks/useCompletion.js'

describe('completionRequestForInput', () => {
  it('routes real slash commands to slash completion', () => {
    expect(completionRequestForInput('/help')).toMatchObject({
      method: 'complete.slash',
      params: { text: '/help' },
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

describe('completionParams', () => {
  it('attaches the session id to path completion, not only slash completion', () => {
    const path = completionRequestForInput('./')!
    const slash = completionRequestForInput('/help')!
    expect(completionParams(path, 'sess-1')).toEqual({ word: './', session_id: 'sess-1' })
    expect(completionParams(slash, 'sess-1')).toEqual({ text: '/help', session_id: 'sess-1' })
  })

  it('sends bare params before a session exists', () => {
    const path = completionRequestForInput('./')!
    expect(completionParams(path, null)).toEqual({ word: './' })
  })
})
