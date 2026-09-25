import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { Spinner } from '../components/thinking.js'
import { STATIC_BUSY_GLYPH } from '../lib/motion.js'

// `display.reduced_motion` (port of openai/codex#46040): a screen reader
// re-announces the status line on every repaint, so a busy indicator must
// render ONE static glyph and arm NO frame timer. The animated default is the
// control case — the same component with the flag off still spins.

const mounted: Array<() => void> = []

const mount = (tree: React.ReactElement) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 10 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(tree, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  mounted.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return () => stripAnsi(output)
}

const BRAILLE = /[\u2800-\u28ff]/

afterEach(() => {
  while (mounted.length) {
    mounted.pop()?.()
  }

  resetUiState()
  vi.restoreAllMocks()
})

describe('reduced motion', () => {
  it('renders a static glyph and arms no frame timer for the thinking spinner', () => {
    patchUiState({ reducedMotion: true })

    const intervals = vi.spyOn(globalThis, 'setInterval')
    const output = mount(<Spinner color="white" />)

    expect(output()).toContain(STATIC_BUSY_GLYPH)
    expect(output()).not.toMatch(BRAILLE)
    expect(intervals).not.toHaveBeenCalled()
  })

  it('keeps the animated braille spinner when the flag is off', () => {
    patchUiState({ reducedMotion: false })

    const intervals = vi.spyOn(globalThis, 'setInterval')
    const output = mount(<Spinner color="white" />)

    expect(output()).toMatch(BRAILLE)
    expect(intervals).toHaveBeenCalled()
  })
})
