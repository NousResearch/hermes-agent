import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync, mkdirSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { resolvePython } from './gatewayClient.js'

describe('resolvePython (exported for direct testing — #63754)', () => {
  const originalEnv = { ...process.env }

  beforeEach(() => {
    delete process.env.HERMES_PYTHON
    delete process.env.PYTHON
    delete process.env.VIRTUAL_ENV
  })

  afterEach(() => {
    for (const k of Object.keys(process.env)) {
      if (!(k in originalEnv)) delete process.env[k]
    }
    Object.assign(process.env, originalEnv)
  })

  it('honors HERMES_PYTHON when set', () => {
    process.env.HERMES_PYTHON = '/custom/path/to/python'
    expect(resolvePython('/any/root')).toBe('/custom/path/to/python')
  })

  it('honors PYTHON as a fallback to HERMES_PYTHON', () => {
    process.env.PYTHON = '/custom/python'
    expect(resolvePython('/any/root')).toBe('/custom/python')
  })

  it('falls back to venv bin/python when VIRTUAL_ENV is set and the file exists', () => {
    const dir = mkdtempSync(join(tmpdir(), 'rp-venv-'))
    const bin = join(dir, 'bin')
    mkdirSync(bin, { recursive: true })
    const py = join(bin, 'python')
    writeFileSync(py, '#!/bin/sh\necho py\n')
    chmodSync(py, 0o755)

    process.env.VIRTUAL_ENV = dir

    expect(resolvePython('/any/root')).toBe(py)

    rmSync(dir, { recursive: true, force: true })
  })

  it('returns a string when no env vars are set (does not throw with undefined root)', () => {
    // Regression for #63754: import.meta.dirname was undefined on Node < 20.11,
    // which propagated through `resolvePython(root)` to `path.resolve(undefined, ...)`
    // and crashed with ERR_INVALID_ARG_TYPE.
    let result: string
    expect(() => {
      // Simulate the pre-fix bug: call with the literal value that
      // import.meta.dirname would have produced — undefined — and assert
      // resolvePython does not throw and returns a usable string.
      result = resolvePython(undefined as unknown as string)
    }).not.toThrow()
    expect(typeof result).toBe('string')
    expect(result.length).toBeGreaterThan(0)
  })

  it('returns a string when root is an empty string', () => {
    let result: string
    expect(() => {
      result = resolvePython('')
    }).not.toThrow()
    expect(typeof result).toBe('string')
    expect(result.length).toBeGreaterThan(0)
  })

  it('prefers VIRTUAL_ENV bin/python over root-based discovery', () => {
    const venvDir = mkdtempSync(join(tmpdir(), 'rp-venv-pref-'))
    const venvBin = join(venvDir, 'bin')
    mkdirSync(venvBin, { recursive: true })
    const venvPy = join(venvBin, 'python')
    writeFileSync(venvPy, '#!/bin/sh\necho venv\n')
    chmodSync(venvPy, 0o755)

    const rootDir = mkdtempSync(join(tmpdir(), 'rp-root-pref-'))
    const rootBin = join(rootDir, '.venv', 'bin')
    mkdirSync(rootBin, { recursive: true })
    const rootPy = join(rootBin, 'python')
    writeFileSync(rootPy, '#!/bin/sh\necho root\n')
    chmodSync(rootPy, 0o755)

    process.env.VIRTUAL_ENV = venvDir

    expect(resolvePython(rootDir)).toBe(venvPy)

    rmSync(venvDir, { recursive: true, force: true })
    rmSync(rootDir, { recursive: true, force: true })
  })
})
