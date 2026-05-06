import { writeSync } from 'fs'
import { Readable } from 'node:stream'

import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { renderSync } from '../root.js'
import { CURSOR_HOME, cursorPosition, ERASE_SCREEN, ERASE_SCROLLBACK } from '../termio/csi.js'
import { ENTER_ALT_SCREEN, EXIT_ALT_SCREEN } from '../termio/dec.js'

import { AlternateScreen } from './AlternateScreen.js'
import Box from './Box.js'
import Text from './Text.js'

vi.mock('fs', async () => {
  const actual = await vi.importActual<typeof import('fs')>('fs')

  return { ...actual, writeSync: vi.fn(() => 0) }
})

class TestStdin extends Readable {
  isRaw = false
  isTTY = true
  _read() {}
  setRawMode(value: boolean) {
    this.isRaw = value

    return this
  }
}

const stub = (key: 'columns' | 'isTTY' | 'rows', value: number | boolean) => {
  const prev = Object.getOwnPropertyDescriptor(process.stdout, key)
  Object.defineProperty(process.stdout, key, { configurable: true, value })

  return () => {
    if (prev) {
      Object.defineProperty(process.stdout, key, prev)
    } else {
      delete (process.stdout as unknown as Record<string, unknown>)[key]
    }
  }
}

const ROWS = 12

describe('AlternateScreen', () => {
  let restore: Array<() => void>
  let stream: ReturnType<typeof vi.fn>
  let sync: ReturnType<typeof vi.fn>

  beforeEach(() => {
    restore = [stub('columns', 40), stub('rows', ROWS), stub('isTTY', true)]
    stream = vi.fn(() => true)
    sync = vi.fn(() => 0)
    vi.spyOn(process.stdout, 'write').mockImplementation(stream as typeof process.stdout.write)
    vi.mocked(writeSync).mockImplementation(sync as unknown as typeof writeSync)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    restore.splice(0).forEach(fn => fn())
  })

  const mount = () =>
    renderSync(
      React.createElement(
        AlternateScreen,
        null,
        React.createElement(Box, null, React.createElement(Text, null, 'hi'))
      ),
      {
        exitOnCtrlC: false,
        patchConsole: false,
        stdin: new TestStdin() as NodeJS.ReadStream,
        stdout: process.stdout
      }
    )

  const streamBytes = () => stream.mock.calls.map(args => String(args[0])).join('')
  const syncBytes = () => sync.mock.calls.map(args => String(args[1])).join('')

  it('enters alt screen without clearing host scrollback', () => {
    const app = mount()
    const mounted = streamBytes()

    expect(mounted).toContain(ENTER_ALT_SCREEN)
    expect(mounted).toContain(ERASE_SCREEN)
    expect(mounted).toContain(CURSOR_HOME)
    expect(mounted).not.toContain(ERASE_SCROLLBACK)

    app.unmount()
  })

  it('does not queue alt-screen content on the buffered stream during unmount', () => {
    // Buffered process.stdout.write may flush AFTER the synchronous
    // writeSync(1, EXIT_ALT_SCREEN). Anything alt-screen-shaped queued on
    // the stream during unmount — alt-screen exit, the per-frame
    // cursorPosition(rows, 1) park patch, screen erases — therefore lands
    // on the *main* screen and parks the cursor at the bottom, eating the
    // parent shell's resume hint.
    const app = mount()
    stream.mockClear()
    sync.mockClear()

    app.unmount()

    const buffered = streamBytes()
    const synced = syncBytes()

    expect(synced).toContain(EXIT_ALT_SCREEN)
    expect(buffered).not.toContain(EXIT_ALT_SCREEN)
    expect(buffered).not.toContain(ENTER_ALT_SCREEN)
    expect(buffered).not.toContain(cursorPosition(ROWS, 1))
    expect(buffered).not.toContain(ERASE_SCREEN)
    expect(buffered).not.toContain(ERASE_SCROLLBACK)
  })
})
