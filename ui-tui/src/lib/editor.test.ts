import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

import { beforeEach, describe, expect, it } from 'vitest'

import { resolveEditor } from './editor.js'

const exe = (dir: string, name: string): string => {
  const path = join(dir, name)

  writeFileSync(path, '#!/bin/sh\nexit 0\n')
  chmodSync(path, 0o755)

  return path
}

describe('resolveEditor', () => {
  let dir: string

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'editor-test-'))
  })

  it('honors $VISUAL above all else', () => {
    expect(resolveEditor({ EDITOR: 'vim', PATH: dir, VISUAL: 'helix' })).toEqual(['helix'])
  })

  it('falls back to $EDITOR when $VISUAL is unset', () => {
    expect(resolveEditor({ EDITOR: 'nvim', PATH: dir })).toEqual(['nvim'])
  })

  it('shell-tokenizes editors with arguments', () => {
    expect(resolveEditor({ EDITOR: 'code --wait', PATH: dir })).toEqual(['code', '--wait'])
    expect(resolveEditor({ PATH: dir, VISUAL: 'emacsclient -t' })).toEqual(['emacsclient', '-t'])
  })

  it('adds --wait to explicit VS Code editor commands so Hermes sees the saved prompt', () => {
    expect(resolveEditor({ EDITOR: 'code', PATH: dir })).toEqual(['code', '--wait'])
    expect(resolveEditor({ EDITOR: 'code --reuse-window', PATH: dir })).toEqual(['code', '--reuse-window', '--wait'])
    expect(resolveEditor({ EDITOR: 'code-insiders --wait', PATH: dir })).toEqual(['code-insiders', '--wait'])
  })

  it('ignores whitespace-only env vars', () => {
    const expected = exe(dir, 'editor')

    expect(resolveEditor({ EDITOR: '   ', PATH: dir, VISUAL: '' })).toEqual([expected])
  })

  it('prefers `editor` over nano over vi on $PATH', () => {
    exe(dir, 'nano')
    exe(dir, 'vi')
    const expected = exe(dir, 'editor')

    expect(resolveEditor({ PATH: dir })).toEqual([expected])
  })

  it('uses VS Code with --wait as the default fallback when code is on $PATH', () => {
    const expected = exe(dir, 'code')
    exe(dir, 'editor')

    expect(resolveEditor({ PATH: dir })).toEqual([expected, '--wait'])
  })

  it('falls back to nano before vi when both exist', () => {
    exe(dir, 'vi')
    const expected = exe(dir, 'nano')

    expect(resolveEditor({ PATH: dir })).toEqual([expected])
  })

  it('returns ["vi"] when $PATH is empty', () => {
    expect(resolveEditor({ PATH: '' })).toEqual(['vi'])
  })

  it('walks multi-entry $PATH', () => {
    const a = mkdtempSync(join(tmpdir(), 'editor-a-'))
    const b = mkdtempSync(join(tmpdir(), 'editor-b-'))
    const expected = exe(b, 'editor')

    expect(resolveEditor({ PATH: [a, b].join(delimiter) })).toEqual([expected])
  })

  it('uses notepad.exe on Windows when no env override', () => {
    expect(resolveEditor({ PATH: dir }, 'win32')).toEqual(['notepad.exe'])
  })
})
