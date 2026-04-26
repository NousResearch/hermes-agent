import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { resolveEditor } from './editor.js'

describe('resolveEditor', () => {
  let dir: string

  const exe = (name: string) => {
    const path = join(dir, name)
    writeFileSync(path, '#!/bin/sh\nexit 0\n')
    chmodSync(path, 0o755)

    return path
  }

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'editor-test-'))
  })

  afterEach(() => {
    // tmp dir is small; let the OS reap it
  })

  it('honors $VISUAL above all else', () => {
    expect(resolveEditor({ EDITOR: 'vim', PATH: dir, VISUAL: 'helix' })).toBe('helix')
  })

  it('falls back to $EDITOR when $VISUAL is unset', () => {
    expect(resolveEditor({ EDITOR: 'nvim', PATH: dir })).toBe('nvim')
  })

  it('prefers nvim over vim over vi over nano on $PATH', () => {
    exe('nano')
    exe('vi')
    exe('vim')
    const nvim = exe('nvim')

    expect(resolveEditor({ PATH: dir })).toBe(nvim)
  })

  it('falls back to vi when only vi and nano exist', () => {
    exe('nano')
    const vi = exe('vi')

    expect(resolveEditor({ PATH: dir })).toBe(vi)
  })

  it('returns literal "vi" when nothing on PATH and no env', () => {
    mkdirSync(join(dir, 'empty'), { recursive: true })

    expect(resolveEditor({ PATH: join(dir, 'empty') })).toBe('vi')
  })

  it('walks multi-entry PATH', () => {
    const a = mkdtempSync(join(tmpdir(), 'editor-a-'))
    const b = mkdtempSync(join(tmpdir(), 'editor-b-'))

    writeFileSync(join(b, 'vim'), '#!/bin/sh\n')
    chmodSync(join(b, 'vim'), 0o755)

    expect(resolveEditor({ PATH: [a, b].join(delimiter) })).toBe(join(b, 'vim'))
  })
})
