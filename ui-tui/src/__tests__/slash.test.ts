import { describe, expect, it } from 'vitest'

import { parseSlashCommand } from '../domain/slash.js'

describe('parseSlashCommand', () => {
  it('parses a bare command with no argument', () => {
    expect(parseSlashCommand('/help')).toEqual({ arg: '', cmd: '/help', name: 'help' })
  })

  it('lowercases the command name and keeps a single-line argument', () => {
    expect(parseSlashCommand('/Goal ship the feature')).toEqual({
      arg: 'ship the feature',
      cmd: '/Goal ship the feature',
      name: 'goal'
    })
  })

  it('keeps newlines in a multi-line argument (regression for #41323)', () => {
    // The old split(/\s+/) + join(' ') flattened newlines into spaces, so a
    // pasted multi-line `/goal` lost its line breaks before reaching the agent.
    const cmd = '/goal line one\nline two\nline three'

    expect(parseSlashCommand(cmd)).toEqual({
      arg: 'line one\nline two\nline three',
      cmd,
      name: 'goal'
    })
  })

  it('returns an empty name when there is no command token', () => {
    expect(parseSlashCommand('/')).toEqual({ arg: '', cmd: '/', name: '' })
  })
})
