import { describe, expect, it } from 'vitest'

import { inlineSlashTrigger, modelSlashIntent } from '../domain/slash.js'
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

  it('completes a second slash in a line that starts with a command', () => {
    // Only the first slash is an invocation. Routing the whole line to the
    // completer offered nothing, so `/work /cle` went dead while
    // `do /work then /cle` completed fine.
    expect(completionRequestForInput('/work /cle')).toMatchObject({
      method: 'complete.slash',
      params: { text: '/cle' },
      replaceFrom: 7,
      skillsOnly: true
    })
  })

  it('leaves a command own arguments to the command', () => {
    for (const input of ['/personality alic', '/cron ad', '/details ']) {
      expect(completionRequestForInput(input)).toEqual({
        method: 'complete.slash',
        params: { text: input },
        replaceFrom: 1
      })
    }
  })

  it('routes a real mid-message path to path completion, not skills', () => {
    expect(completionRequestForInput('open src/foo/ba')).toMatchObject({ method: 'complete.path' })
    expect(completionRequestForInput('open /usr/lo')).toMatchObject({ method: 'complete.path' })
  })
})

describe('modelSlashIntent', () => {
  it('opens the hop overlay for a bare /model', () => {
    expect(modelSlashIntent('')).toEqual({ type: 'overlay' })
    expect(modelSlashIntent('--refresh')).toEqual({ type: 'overlay', refresh: true })
  })

  it('opens a session-only hop for /model --session', () => {
    expect(modelSlashIntent('--session')).toEqual({ type: 'overlay', sessionOnly: true })
    expect(modelSlashIntent('--session --refresh')).toEqual({
      type: 'overlay',
      refresh: true,
      sessionOnly: true
    })
    expect(modelSlashIntent('--session --provider')).toEqual({
      type: 'overlay',
      stage: 'provider',
      sessionOnly: true
    })
  })

  it('still sets when --session is attached to a model id', () => {
    expect(modelSlashIntent('hermes-4 --session')).toEqual({ type: 'set' })
  })

  it('does not hide persist when --global is also present', () => {
    expect(modelSlashIntent('--session --global')).toEqual({ type: 'overlay' })
    expect(modelSlashIntent('--global --session')).toEqual({ type: 'overlay' })
  })

  it('opens the provider list when --provider has no slug', () => {
    expect(modelSlashIntent('--provider')).toEqual({ type: 'overlay', stage: 'provider' })
    expect(modelSlashIntent('--provider --refresh')).toEqual({
      type: 'overlay',
      refresh: true,
      stage: 'provider'
    })
  })

  it('sets the model when a positional id is present', () => {
    expect(modelSlashIntent('claude-sonnet-4.6 --provider nous')).toEqual({ type: 'set' })
  })
})

describe('completionRequestForInput — /model args', () => {
  it('asks the slash completer for /model flags and hops', () => {
    expect(completionRequestForInput('/model ')).toEqual({
      method: 'complete.slash',
      params: { text: '/model ' },
      replaceFrom: 1
    })
    expect(completionRequestForInput('/model --pro')).toEqual({
      method: 'complete.slash',
      params: { text: '/model --pro' },
      replaceFrom: 1
    })
  })
})
