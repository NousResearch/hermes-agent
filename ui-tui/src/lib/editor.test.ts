import { chmodSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { openInEditor, resolveEditor } from './editor.js'

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

  it('ignores whitespace-only env vars', () => {
    const expected = exe(dir, 'editor')

    expect(resolveEditor({ EDITOR: '   ', PATH: dir, VISUAL: '' }, 'linux')).toEqual([expected])
  })

  it('prefers `editor` over nano over vi on $PATH', () => {
    exe(dir, 'nano')
    exe(dir, 'vi')
    const expected = exe(dir, 'editor')

    expect(resolveEditor({ PATH: dir }, 'linux')).toEqual([expected])
  })

  it('falls back to nano before vi when both exist', () => {
    exe(dir, 'vi')
    const expected = exe(dir, 'nano')

    expect(resolveEditor({ PATH: dir }, 'linux')).toEqual([expected])
  })

  it('returns ["vi"] when $PATH is empty', () => {
    expect(resolveEditor({ PATH: '' }, 'linux')).toEqual(['vi'])
  })

  it('walks multi-entry $PATH', () => {
    const a = mkdtempSync(join(tmpdir(), 'editor-a-'))
    const b = mkdtempSync(join(tmpdir(), 'editor-b-'))
    const expected = exe(b, 'editor')

    expect(resolveEditor({ PATH: [a, b].join(delimiter) }, 'linux')).toEqual([expected])
  })

  it('uses notepad.exe on Windows when no env override', () => {
    expect(resolveEditor({ PATH: dir }, 'win32')).toEqual(['notepad.exe'])
  })
})

describe('openInEditor', () => {
  const originalVisual = process.env.VISUAL

  afterEach(() => {
    if (originalVisual === undefined) {
      delete process.env.VISUAL
    } else {
      process.env.VISUAL = originalVisual
    }
  })

  it('returns an editor save that becomes visible shortly after the editor exits', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'editor-delayed-save-test-'))
    const editor = join(dir, 'delayed-editor.mjs')

    writeFileSync(
      editor,
      [
        '#!/usr/bin/env node',
        "import { spawn } from 'node:child_process'",
        'const target = process.argv[2]',
        'const writer = spawn(process.execPath, [',
        "  '-e',",
        "  `setTimeout(() => require('node:fs').writeFileSync(process.argv[1], 'edited prompt'), 100)`,",
        '  target',
        "], { detached: true, stdio: 'ignore' })",
        'writer.unref()'
      ].join('\n')
    )
    chmodSync(editor, 0o755)
    process.env.VISUAL = `${process.execPath} ${editor}`

    await expect(openInEditor('initial draft', '.md')).resolves.toBe('edited prompt')
  })

  it('returns the last stable value when an editor flushes multiple saves', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'editor-multiple-save-test-'))
    const editor = join(dir, 'multi-save-editor.mjs')

    writeFileSync(
      editor,
      [
        '#!/usr/bin/env node',
        "import { spawn } from 'node:child_process'",
        'const target = process.argv[2]',
        'const writer = spawn(process.execPath, [',
        "  '-e',",
        "  `const fs = require('node:fs'); setTimeout(() => fs.writeFileSync(process.argv[1], 'intermediate'), 50); setTimeout(() => fs.writeFileSync(process.argv[1], 'final prompt'), 150)`,",
        '  target',
        "], { detached: true, stdio: 'ignore' })",
        'writer.unref()'
      ].join('\n')
    )
    chmodSync(editor, 0o755)
    process.env.VISUAL = `${process.execPath} ${editor}`

    await expect(openInEditor('initial draft', '.md')).resolves.toBe('final prompt')
  })

  it('returns an unchanged buffer without waiting for the full save timeout', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'editor-unchanged-test-'))
    const editor = join(dir, 'unchanged-editor.mjs')

    writeFileSync(editor, '#!/usr/bin/env node\nprocess.exit(0)\n')
    chmodSync(editor, 0o755)
    process.env.VISUAL = `${process.execPath} ${editor}`

    const startedAt = Date.now()
    await expect(openInEditor('initial draft', '.md')).resolves.toBe('initial draft')
    expect(Date.now() - startedAt).toBeLessThan(1_000)
  })

  it('returns null when the editor exits unsuccessfully', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'editor-failure-test-'))
    const editor = join(dir, 'failing-editor.mjs')

    writeFileSync(editor, '#!/usr/bin/env node\nprocess.exit(1)\n')
    chmodSync(editor, 0o755)
    process.env.VISUAL = `${process.execPath} ${editor}`

    await expect(openInEditor('initial draft', '.md')).resolves.toBeNull()
  })
})
