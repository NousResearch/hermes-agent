import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

import { beforeEach, describe, expect, it } from 'vitest'

import { resolveEditor } from './editor.js'

/**
 * The POSIX-fallback PATH-walk branch only runs when `platform !== 'win32'`,
 * so every fallback test pins `platform: 'linux'` explicitly — that is the
 * code under test, independent of the OS running the suite. The Windows
 * branch has its own dedicated test below (notepad.exe).
 *
 * The `exe` fixture creates a file that `accessSync(X_OK)` accepts on every
 * platform: an extensionless executable (the POSIX shape resolveEditor
 * searches for) that is also chmod'ed — Windows' access() reports X_OK true
 * for every readable file, so the chmod is what makes the fixture honest on
 * POSIX, and the plain file satisfies it on Windows. (A PATHEXT-suffixed
 * fixture like `editor.exe` would NOT work: resolveEditor searches bare
 * names only — `join(dir, 'editor')` — so an `.exe` copy would never be
 * found by the code under test.)
 */
const exe = (dir: string, name: string): string => {
  const path = join(dir, name)

  writeFileSync(path, '#!/bin/sh\nexit 0\n')
  chmodSync(path, 0o755)

  return path
}

const POSIX = 'linux' as const

describe('resolveEditor', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'editor-test-'))
  })

  it('honors $VISUAL above all else', () => {
    expect(resolveEditor({ EDITOR: 'vim', PATH: dir, VISUAL: 'helix' }, POSIX)).toEqual(['helix'])
  })

  it('falls back to $EDITOR when $VISUAL is unset', () => {
    expect(resolveEditor({ EDITOR: 'nvim', PATH: dir }, POSIX)).toEqual(['nvim'])
  })

  it('shell-tokenizes editors with arguments', () => {
    expect(resolveEditor({ EDITOR: 'code --wait', PATH: dir }, POSIX)).toEqual(['code', '--wait'])
    expect(resolveEditor({ PATH: dir, VISUAL: 'emacsclient -t' }, POSIX)).toEqual([
      'emacsclient',
      '-t',
    ])
  })

  it('ignores whitespace-only env vars', () => {
    const expected = exe(dir, 'editor')

    expect(resolveEditor({ EDITOR: '   ', PATH: dir, VISUAL: '' }, POSIX)).toEqual([expected])
  })

  it('prefers `editor` over nano over vi on $PATH', () => {
    exe(dir, 'nano')
    exe(dir, 'vi')
    const expected = exe(dir, 'editor')

    expect(resolveEditor({ PATH: dir }, POSIX)).toEqual([expected])
  })

  it('falls back to nano before vi when both exist', () => {
    exe(dir, 'vi')
    const expected = exe(dir, 'nano')

    expect(resolveEditor({ PATH: dir }, POSIX)).toEqual([expected])
  })

  it('returns ["vi"] when $PATH is empty', () => {
    expect(resolveEditor({ PATH: '' }, POSIX)).toEqual(['vi'])
  })

  it('walks multi-entry $PATH', () => {
    const a = mkdtempSync(join(tmpdir(), 'editor-a-'))
    const b = mkdtempSync(join(tmpdir(), 'editor-b-'))
    const expected = exe(b, 'editor')

    expect(resolveEditor({ PATH: [a, b].join(delimiter) }, POSIX)).toEqual([expected])
  })

  it('uses notepad.exe on Windows when no env override', () => {
    expect(resolveEditor({ PATH: dir }, 'win32')).toEqual(['notepad.exe'])
  })
})
