import { describe, expect, it } from 'vitest'

import { completionToApplyOnEnter } from '../domain/slash.js'

describe('completionToApplyOnEnter', () => {
  const input = 'I want to modify the files @two-agent'
  const compReplace = input.indexOf('@') // 25 — position of '@'

  it('path completion: Enter ALWAYS selects the highlighted row, never falls through to submit', () => {
    const row = '@folder:two-agent-slide/'

    const next = completionToApplyOnEnter(input, row, compReplace, 'path')

    expect(next).toBe('I want to modify the files @folder:two-agent-slide/')
  })

  it('path completion: selects even when the row is a bare name', () => {
    const next = completionToApplyOnEnter(input, 'two-agent-slide', compReplace, 'path')

    expect(next).toBe('I want to modify the files two-agent-slide')
  })

  it('path completion: a row with no text returns null (nothing to select)', () => {
    expect(completionToApplyOnEnter(input, '', compReplace, 'path')).toBeNull()
  })

  it('slash completion keeps original behaviour: Enter accepts when it changes the token', () => {
    const next = completionToApplyOnEnter('/hel', '/help', 1, 'slash')

    expect(next).toBe('/help')
  })

  it('slash completion: whitespace-only delta falls through to submit for already-complete commands', () => {
    // trailing space the gateway adds after `/exit` — must NOT swallow the Enter
    expect(completionToApplyOnEnter('/exit', '/exit ', 1, 'slash')).toBeNull()
  })

  it('slash completion: null kind (no active dropdown) keeps slash semantics', () => {
    expect(completionToApplyOnEnter('/exit', '/exit ', 1, null)).toBeNull()
  })
})
