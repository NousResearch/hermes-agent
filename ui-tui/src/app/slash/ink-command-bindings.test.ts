import { describe, expect, it } from 'vitest'

import {
  projectInkClientCommandBindings,
  resolveInkClientCommandBinding
} from './ink-command-bindings.js'
import { SLASH_COMMANDS } from './registry.js'

describe('Ink client command bindings', () => {
  it('projects the complete current static handler surface for v1 mixed-version peers', () => {
    const bindings = projectInkClientCommandBindings(SLASH_COMMANDS.map(command => ({ name: `/${command.name}` })))

    expect(bindings).toHaveLength(SLASH_COMMANDS.length)
    expect(new Set(bindings.map(binding => binding.canonicalName)).size).toBe(bindings.length)
  })

  it('uses command_id from v2 catalogs and the canonical name for mixed-version fallback', () => {
    expect(
      resolveInkClientCommandBinding({
        command_id: 'ink.fortune',
        execution_owner: 'client',
        name: '/fortune'
      })
    ).toMatchObject({ canonicalName: '/fortune', commandId: 'ink.fortune' })
    expect(resolveInkClientCommandBinding({ name: '/fortune' })).toMatchObject({
      canonicalName: '/fortune',
      commandId: '/fortune'
    })
  })

  it('canonicalizes aliases through the existing registry instead of owning another alias table', () => {
    const mouse = resolveInkClientCommandBinding({
      command_id: 'ink.mouse',
      execution_owner: 'client',
      name: '/mouse'
    })
    const scroll = resolveInkClientCommandBinding({
      command_id: 'ink.mouse',
      execution_owner: 'client',
      name: '/scroll'
    })

    expect(scroll).toMatchObject({ canonicalName: '/mouse', commandId: 'ink.mouse' })
    expect(scroll?.run).toBe(mouse?.run)
  })

  it('does not claim explicitly server-, plugin-, skill-, or agent-owned commands', () => {
    for (const execution_owner of ['server', 'plugin', 'skill', 'agent_turn']) {
      expect(resolveInkClientCommandBinding({ execution_owner, name: '/fortune' })).toBeNull()
    }
  })

  it('does not invent bindings for unknown catalog rows', () => {
    expect(resolveInkClientCommandBinding({ execution_owner: 'client', name: '/does-not-exist' })).toBeNull()
    expect(resolveInkClientCommandBinding({ execution_owner: 'client', name: '   ' })).toBeNull()
  })

  it('returns immutable bindings and projections', () => {
    const binding = resolveInkClientCommandBinding({ execution_owner: 'client', name: '/fortune' })
    const bindings = projectInkClientCommandBindings([{ execution_owner: 'client', name: '/fortune' }])

    expect(Object.isFrozen(binding)).toBe(true)
    expect(Object.isFrozen(bindings)).toBe(true)
  })

  it('fails closed on duplicate stable identities, including case-only variants', () => {
    expect(() =>
      projectInkClientCommandBindings([
        { command_id: 'ink.shared', execution_owner: 'client', name: '/fortune' },
        { command_id: 'INK.SHARED', execution_owner: 'client', name: '/mouse' }
      ])
    ).toThrow('Ink command_id collision')
  })

  it('fails closed when aliases project the same canonical command twice', () => {
    expect(() =>
      projectInkClientCommandBindings([
        { command_id: 'ink.mouse', execution_owner: 'client', name: '/mouse' },
        { command_id: 'ink.scroll', execution_owner: 'client', name: '/scroll' }
      ])
    ).toThrow('Ink canonical command collision')
  })

  it('rejects blank explicit command ids and ownership instead of guessing', () => {
    expect(() =>
      resolveInkClientCommandBinding({ command_id: '   ', execution_owner: 'client', name: '/fortune' })
    ).toThrow('empty command_id')
    expect(() => resolveInkClientCommandBinding({ execution_owner: '   ', name: '/fortune' })).toThrow(
      'empty execution_owner'
    )
  })
})
