import { execFileSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { findSlashCommand, SLASH_COMMANDS } from '../app/slash/registry.js'
import { writeClipboardText } from '../lib/clipboard.js'
import type { SlashRunCtx } from '../app/slash/types.js'

vi.mock('../lib/clipboard.js', () => ({
  writeClipboardText: vi.fn(() => Promise.resolve(true))
}))

type CommandRoute = 'fallback' | 'local' | 'native'

interface CommandRegistryLoad {
  error?: string
  names: string[]
}

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

const loadCommandRegistryNames = (): CommandRegistryLoad => {
  const here = dirname(fileURLToPath(import.meta.url))

  try {
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

    return { names: [...new Set(names)] }
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : String(error),
      names: []
    }
  }
}

const commandRegistry = loadCommandRegistryNames()
const registryIt = commandRegistry.error ? it.skip : it
const skipReason = commandRegistry.error ? commandRegistry.error.split('\n')[0] : ''

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
  beforeEach(() => {
    vi.clearAllMocks()
  })

  if (commandRegistry.error) {
    it.skip(`Python command registry unavailable: ${skipReason}`, () => {})
  }

  registryIt('classifies each command registry command as local/native/fallback', () => {
    const routes = Object.fromEntries(commandRegistry.names.map(name => [name, classifyRoute(name)]))

    expect(routes['model']).toBe('local')
    expect(routes['browser']).toBe('native')
    expect(routes['reload-mcp']).toBe('native')
    expect(routes['rollback']).toBe('native')
    expect(routes['stop']).toBe('native')
  })

  registryIt('keeps every mutating command off slash-worker fallback', () => {
    const routes = Object.fromEntries(commandRegistry.names.map(name => [name, classifyRoute(name)]))

    for (const name of MUTATING_COMMANDS) {
      expect(routes[name], `missing command in registry: ${name}`).toBeDefined()
      expect(routes[name], `mutating command must not fallback: ${name}`).not.toBe('fallback')
    }
  })

  it('/q alias resolves to queue, not quit (#31983)', () => {
    // Regression for #31983: the TUI `quit` command used to carry alias `q`,
    // which collided with the Python-side `/queue` alias. TUI-local commands
    // dispatch before the backend, so `/q` resolved to /quit (session.die)
    // instead of queueing a prompt.
    const cmd = findSlashCommand('q')
    expect(cmd, '/q must resolve to a command').toBeDefined()
    expect(cmd!.name).toBe('queue')
  })

  it('/copy number selects the nth-latest assistant response', () => {
    const cmd = findSlashCommand('copy')
    expect(cmd, '/copy must resolve to a command').toBeDefined()

    const sysMessages: string[] = []
    const history = [
      { role: 'assistant', text: 'one' },
      { role: 'user', text: 'ignore me' },
      { role: 'assistant', text: 'two' },
      { role: 'assistant', text: 'three' }
    ]
    const ctx = {
      composer: { hasSelection: false },
      local: { getHistoryItems: () => history },
      stale: () => false,
      transcript: { sys: (message: string) => sysMessages.push(message) }
    } as unknown as SlashRunCtx

    cmd!.run('2', ctx, 'copy')

    expect(writeClipboardText).toHaveBeenCalledWith('two')
    expect(writeClipboardText).not.toHaveBeenCalledWith('one')
    expect(writeClipboardText).not.toHaveBeenCalledWith('three')
  })
})
