import { execFileSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { SLASH_COMMANDS } from '../app/slash/registry.js'

type CommandRoute = 'fallback' | 'local' | 'native'

const NATIVE_MUTATING_COMMANDS = new Set(['browser', 'busy', 'fast', 'reload-mcp', 'rollback', 'stop'])

const MUTATING_COMMANDS = [
  'background',
  'branch',
  'browser',
  'busy',
  'clear',
  'compress',
  'fast',
  'model',
  'new',
  'personality',
  'queue',
  'reasoning',
  'reload-mcp',
  'retry',
  'rollback',
  'steer',
  'stop',
  'title',
  'tools',
  'undo',
  'verbose',
  'voice',
  'yolo'
] as const

const loadCommandRegistryNames = (): string[] => {
  const here = dirname(fileURLToPath(import.meta.url))

  const names = JSON.parse(
    execFileSync(
      process.env.PYTHON ?? 'python3',
      [
        '-c',
        'import json; from hermes_cli.commands import COMMAND_REGISTRY; print(json.dumps([c.name for c in COMMAND_REGISTRY]))'
      ],
      { cwd: resolve(here, '../../..'), encoding: 'utf8' }
    )
  ) as string[]

  return [...new Set(names)]
}

const LOCAL_COMMAND_NAMES = new Set(
  SLASH_COMMANDS.flatMap(command => [command.name, ...(command.aliases ?? [])].map(name => name.toLowerCase()))
)

const classifyRoute = (name: string): CommandRoute => {
  const normalized = name.toLowerCase()

  if (NATIVE_MUTATING_COMMANDS.has(normalized)) {
    return 'native'
  }

  if (LOCAL_COMMAND_NAMES.has(normalized)) {
    return 'local'
  }

  return 'fallback'
}

describe('slash parity matrix', () => {
  it('classifies each command registry command as local/native/fallback', () => {
    const routes = Object.fromEntries(loadCommandRegistryNames().map(name => [name, classifyRoute(name)]))

    expect(routes['model']).toBe('local')
    expect(routes['browser']).toBe('native')
    expect(routes['reload-mcp']).toBe('native')
    expect(routes['rollback']).toBe('native')
    expect(routes['stop']).toBe('native')
  })

  it('keeps every mutating command off slash-worker fallback', () => {
    const routes = Object.fromEntries(loadCommandRegistryNames().map(name => [name, classifyRoute(name)]))

    for (const name of MUTATING_COMMANDS) {
      expect(routes[name], `missing command in registry: ${name}`).toBeDefined()
      expect(routes[name], `mutating command must not fallback: ${name}`).not.toBe('fallback')
    }
  })
})
