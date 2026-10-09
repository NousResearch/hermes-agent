import { describe, expect, it } from 'vitest'

import { inlineSlashTrigger, splitSlashSkillRefs } from '../domain/slash.js'
import { completionRequestForInput } from '../hooks/useCompletion.js'

describe('inlineSlashTrigger', () => {
  it('detects a slash typed mid-message', () => {
    // The reported bug: only a position-0 slash offered anything, so
    // "please run /cle" completed nothing.
    expect(inlineSlashTrigger('please run /cle')).toEqual({ query: 'cle', start: 11 })
  })

  it('detects a bare slash after whitespace, before any name is typed', () => {
    expect(inlineSlashTrigger('please run /')).toEqual({ query: '', start: 11 })
  })

  it('fires after a newline, not just a space', () => {
    expect(inlineSlashTrigger('text\n/skill')).toEqual({ query: 'skill', start: 5 })
  })

  it('does not fire at position 0 — that is a command invocation', () => {
    expect(inlineSlashTrigger('/clean')).toBeNull()
    expect(inlineSlashTrigger('/')).toBeNull()
  })

  it('leaves file paths alone', () => {
    expect(inlineSlashTrigger('look at /usr/local/bin')).toBeNull()
    expect(inlineSlashTrigger('check src/foo/bar')).toBeNull()
    expect(inlineSlashTrigger('and/or')).toBeNull()
  })

  it('stops at the command token — an inline reference takes no args', () => {
    // Only a position-0 slash is a real invocation, so `/personality alic`
    // mid-message is prose with a reference in it, already ended.
    expect(inlineSlashTrigger('hello there /personality alic')).toBeNull()
  })

  it('reports a start index that replaces only the typed token', () => {
    const text = 'please run /cle'
    const trigger = inlineSlashTrigger(text)!

    expect(text.slice(0, trigger.start)).toBe('please run ')
    expect(text.slice(trigger.start)).toBe('/cle')
  })
})

describe('completionRequestForInput — inline skill references', () => {
  it('asks for skills only when the slash is mid-message', () => {
    const request = completionRequestForInput('please run /cle')

    expect(request).toMatchObject({
      method: 'complete.slash',
      params: { text: '/cle' },
      replaceFrom: 12,
      skillsOnly: true
    })
  })

  it('keeps the full command set at position 0', () => {
    expect(completionRequestForInput('/cle')).toEqual({
      method: 'complete.slash',
      params: { text: '/cle' },
      replaceFrom: 1
    })
  })

  it('routes a real mid-message path to path completion, not skills', () => {
    expect(completionRequestForInput('open src/foo/ba')).toMatchObject({ method: 'complete.path' })
    expect(completionRequestForInput('open /usr/lo')).toMatchObject({ method: 'complete.path' })
  })
})

describe('splitSlashSkillRefs', () => {
  it('marks a skill referenced mid-prose', () => {
    expect(splitSlashSkillRefs('clean this up with /clean')).toEqual([
      { ref: false, text: 'clean this up with ' },
      { ref: true, text: '/clean' }
    ])
  })

  it('keeps the prose on both sides of the reference', () => {
    expect(splitSlashSkillRefs('run /clean then ship')).toEqual([
      { ref: false, text: 'run ' },
      { ref: true, text: '/clean' },
      { ref: false, text: ' then ship' }
    ])
  })

  it('does not mark paths', () => {
    for (const text of ['look at /usr/local/bin', 'check src/foo/bar', 'a 3 /4 b']) {
      expect(splitSlashSkillRefs(text)).toEqual([{ ref: false, text }])
    }
  })

  it('does not mark a leading slash — that is a command, not a reference', () => {
    expect(splitSlashSkillRefs('/clean')).toEqual([{ ref: false, text: '/clean' }])
  })

  it('round-trips the input exactly', () => {
    for (const text of ['run /clean then /work ok', 'plain text', '', 'look at /usr/local/bin']) {
      expect(
        splitSlashSkillRefs(text)
          .map(s => s.text)
          .join('')
      ).toBe(text)
    }
  })

  it('always returns at least one segment', () => {
    expect(splitSlashSkillRefs('')).toEqual([{ ref: false, text: '' }])
  })
})
