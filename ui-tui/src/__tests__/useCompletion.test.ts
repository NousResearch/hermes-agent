import { describe, expect, it } from 'vitest'

import { completionRequestForInput, getLocalSlashCompletion } from '../hooks/useCompletion.js'
import type { SlashCatalog } from '../types.js'

const catalog: SlashCatalog = {
  canon: {
    '/bg': '/background',
    '/background': '/background',
    '/help': '/help',
    '/model': '/model',
    '/new': '/new',
    '/reasoning': '/reasoning',
    '/reset': '/new'
  },
  categories: [],
  pairs: [
    ['/new', 'Start a new session'],
    ['/background', 'Run a prompt in the background'],
    ['/reasoning', 'Manage reasoning effort and display'],
    ['/help', 'Show available commands'],
    ['/model', 'Switch model for this session'],
    ['/compact', 'Toggle compact display mode'],
    ['/gif-search', 'Search for GIFs across providers']
  ],
  skillCount: 1,
  sub: {
    '/reasoning': ['none', 'low', 'high', 'show', 'hide']
  }
}

describe('getLocalSlashCompletion', () => {
  it('completes cached top-level slash commands and aliases locally', () => {
    expect(getLocalSlashCompletion('/he', catalog)).toEqual({
      items: [{ display: '/help', meta: 'Show available commands', text: 'help' }],
      replace_from: 1
    })

    expect(getLocalSlashCompletion('/re', catalog)).toEqual({
      items: [
        {
          display: '/reset',
          meta: 'Start a new session (alias for /new)',
          text: 'reset'
        },
        {
          display: '/reasoning',
          meta: 'Manage reasoning effort and display',
          text: 'reasoning'
        }
      ],
      replace_from: 1
    })
  })

  it('adds a trailing space for exact local command matches', () => {
    expect(getLocalSlashCompletion('/help', catalog)).toEqual({
      items: [{ display: '/help', meta: 'Show available commands', text: 'help ' }],
      replace_from: 1
    })

    expect(getLocalSlashCompletion('/gif-search', catalog)).toEqual({
      items: [{ display: '/gif-search', meta: 'Search for GIFs across providers', text: 'gif-search ' }],
      replace_from: 1
    })
  })

  it('completes static subcommands locally', () => {
    expect(getLocalSlashCompletion('/reasoning sh', catalog)).toEqual({
      items: [{ display: 'show', text: 'show' }],
      replace_from: 11
    })

    expect(getLocalSlashCompletion('/reasoning show', catalog)).toEqual({
      items: [],
      replace_from: 11
    })
  })

  it('falls back to the gateway for runtime subcommand sources and unknown slash prefixes', () => {
    expect(getLocalSlashCompletion('/model so', catalog)).toBeNull()
    expect(getLocalSlashCompletion('/plugin', catalog)).toBeNull()
  })
})

describe('completionRequestForInput', () => {
  it('routes real slash commands to slash completion', () => {
    expect(completionRequestForInput('/help')).toMatchObject({
      method: 'complete.slash',
      params: { text: '/help' },
      replaceFrom: 1
    })
  })

  it('leaves model completion to the model picker', () => {
    expect(completionRequestForInput('/model so')).toBeNull()
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
