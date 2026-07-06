import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { logForDebugging } from './debug.js'

describe('logForDebugging', () => {
  let dir: string
  let logPath: string
  let originalEnabled: string | undefined
  let originalPath: string | undefined

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'hermes-ink-debug-test-'))
    logPath = join(dir, 'nested', 'tui-debug.log')
    originalEnabled = process.env.HERMES_INK_DEBUG_LOG
    originalPath = process.env.HERMES_INK_DEBUG_LOG_PATH
    process.env.HERMES_INK_DEBUG_LOG_PATH = logPath
  })

  afterEach(() => {
    if (originalEnabled === undefined) {
      delete process.env.HERMES_INK_DEBUG_LOG
    } else {
      process.env.HERMES_INK_DEBUG_LOG = originalEnabled
    }

    if (originalPath === undefined) {
      delete process.env.HERMES_INK_DEBUG_LOG_PATH
    } else {
      process.env.HERMES_INK_DEBUG_LOG_PATH = originalPath
    }

    rmSync(dir, { recursive: true, force: true })
  })

  it('is a safe no-op and writes nothing when not explicitly enabled', () => {
    delete process.env.HERMES_INK_DEBUG_LOG

    logForDebugging('should not be written anywhere')

    expect(existsSync(logPath)).toBe(false)
  })

  it('appends a timestamped line to the debug log file when enabled', () => {
    process.env.HERMES_INK_DEBUG_LOG = '1'

    logForDebugging('hello from the tui', { level: 'warn' })

    expect(existsSync(logPath)).toBe(true)

    const contents = readFileSync(logPath, 'utf8')

    expect(contents).toContain('[warn] hello from the tui')
    // ISO 8601 timestamp prefix, e.g. 2026-07-06T20:00:00.000Z
    expect(contents).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z /)
  })

  it('appends multiple calls as separate lines rather than overwriting', () => {
    process.env.HERMES_INK_DEBUG_LOG = '1'

    logForDebugging('first message')
    logForDebugging('second message')

    const lines = readFileSync(logPath, 'utf8').trim().split('\n')

    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain('first message')
    expect(lines[1]).toContain('second message')
  })

  it('defaults to level "debug" when no level is given', () => {
    process.env.HERMES_INK_DEBUG_LOG = '1'

    logForDebugging('unlabeled message')

    expect(readFileSync(logPath, 'utf8')).toContain('[debug] unlabeled message')
  })
})
