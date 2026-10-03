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

  it('does not route absolute paths through slash completion', () => {
    expect(
      completionRequestForInput('/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md')
    ).toMatchObject({
      method: 'complete.path',
      params: {
        word: '/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md'
      },
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

  it.each(['/model', '/mode', '/modelx'])('keeps command token %s discoverable', (input) => {
    expect(completionRequestForInput(input)).toMatchObject({
      method: 'complete.slash',
      params: { text: input },
      replaceFrom: 1
    })
  })

  it.each(['/model ', '/model\t', '/model example'])('leaves model arguments %s to the picker', (input) => {
    expect(completionRequestForInput(input)).toBeNull()
  })

  it('restores slash completion after deleting model arguments and their separator', () => {
    for (const input of ['/mode', '/model', '/model ', '/model x', '/model ', '/model', '/mode']) {
      expect(completionRequestForInput(input)?.method ?? null).toBe(
        input.includes(' ') ? null : 'complete.slash'
      )
    }
  })

  it('leaves plain text alone', () => {
    expect(completionRequestForInput('hello there')).toBeNull()
  })
})
