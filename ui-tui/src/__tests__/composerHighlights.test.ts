import { describe, expect, it } from 'vitest'

import { highlightsStable, splitComposerHighlights } from '../domain/composerHighlights.js'

const painted = (text: string) =>
  splitComposerHighlights(text)
    .filter(segment => segment.ref)
    .map(segment => segment.text)

describe('splitComposerHighlights', () => {
  it('marks a command invocation and a skill named mid-prose', () => {
    expect(painted('/work fix the leak')).toEqual(['/work'])
    expect(painted('clean this up with /clean')).toEqual(['/clean'])
    expect(painted('run /clean then /work')).toEqual(['/clean', '/work'])
  })

  it('marks @ references, including quoted values with spaces', () => {
    expect(painted('see @file:src/a.ts please')).toEqual(['@file:src/a.ts'])
    expect(painted('see @file:`my notes.md` please')).toEqual(['@file:`my notes.md`'])
    expect(painted('diff @diff and @staged')).toEqual(['@diff', '@staged'])
  })

  it('marks attachment and paste tokens', () => {
    expect(painted('what is in [[ Image 1 ]] here')).toEqual(['[[ Image 1 ]]'])
    expect(painted('paste [[ log.. [3 lines] ]] ok')).toEqual(['[[ log.. [3 lines] ]]'])
  })

  it('marks every kind in one message', () => {
    expect(painted('/work with @file:a.ts and [[ Image 2 ]]')).toEqual(['/work', '@file:a.ts', '[[ Image 2 ]]'])
  })

  it('leaves paths, bare slashes, and email addresses alone', () => {
    for (const text of [
      'look at /usr/local/bin',
      'check src/foo/bar',
      'a 3 /4 b',
      'either / or',
      'email me@example.com'
    ]) {
      expect(splitComposerHighlights(text)).toEqual([{ ref: false, text }])
    }
  })

  it('marks a half-typed token so the accent tracks the caret', () => {
    // The composer paints while you type — waiting for the token to close
    // would flash the accent on only after the last character. A bare `/`
    // counts at the caret: that's the command menu opening.
    expect(painted('/wor')).toEqual(['/wor'])
    expect(painted('ref @fi')).toEqual(['@fi'])
    expect(painted('/')).toEqual(['/'])
  })

  it('round-trips the input exactly', () => {
    for (const text of ['/work a', 'x @file:b [[ Image 1 ]]', 'plain text', '', 'look at /usr/local/bin']) {
      expect(
        splitComposerHighlights(text)
          .map(segment => segment.text)
          .join('')
      ).toBe(text)
    }
  })

  it('always returns at least one segment', () => {
    expect(splitComposerHighlights('')).toEqual([{ ref: false, text: '' }])
  })
})

describe('highlightsStable', () => {
  // Fast-echo writes ONLY the new cells, so it may run only when every
  // character already on screen keeps the colour it had.
  it('allows the bypass while a token just grows', () => {
    expect(highlightsStable('/wor', '/work')).toBe(true)
    expect(highlightsStable('hello', 'hello ')).toBe(true)
  })

  it('blocks the bypass when a keystroke re-colours existing cells', () => {
    expect(highlightsStable('[[ a ]', '[[ a ]]')).toBe(false)
    expect(highlightsStable('/usr', '/usr/')).toBe(false)
  })
})
