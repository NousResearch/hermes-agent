import { writeSync } from 'fs'
import { Readable } from 'node:stream'

import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import instances from '../instances.js'
import { renderSync } from '../root.js'
import { CURSOR_HOME, ERASE_SCREEN, ERASE_SCROLLBACK } from '../termio/csi.js'
import { ENTER_ALT_SCREEN, EXIT_ALT_SCREEN } from '../termio/dec.js'

import { AlternateScreen } from './AlternateScreen.js'
import Box from './Box.js'
import Text from './Text.js'

vi.mock('fs', async () => {
  const actual = await vi.importActual<typeof import('fs')>('fs')

  return {
    ...actual,
    writeSync: vi.fn(() => 0)
  }
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

const replaceStdoutProp = (key: 'columns' | 'isTTY' | 'rows', value: number | boolean) => {
  const descriptor = Object.getOwnPropertyDescriptor(process.stdout, key)

  Object.defineProperty(process.stdout, key, { configurable: true, value })

  return () => {
    if (descriptor) {
      Object.defineProperty(process.stdout, key, descriptor)
    } else {
      delete (process.stdout as unknown as Record<string, unknown>)[key]
    }
  }
}

const count = (value: string, needle: string) => value.split(needle).length - 1

describe('AlternateScreen', () => {
  const restoreStdoutProps: Array<() => void> = []
  let stdoutWrite: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    restoreStdoutProps.push(
      replaceStdoutProp('columns', 40),
      replaceStdoutProp('rows', 12),
      replaceStdoutProp('isTTY', true)
    )
    stdoutWrite = vi.spyOn(process.stdout, 'write').mockImplementation((() => true) as typeof process.stdout.write)
    vi.mocked(writeSync).mockClear()
  })

  afterEach(() => {
    stdoutWrite.mockRestore()

    while (restoreStdoutProps.length) {
      restoreStdoutProps.pop()?.()
    }
  })

  const renderAltScreen = () =>
    renderSync(
      React.createElement(AlternateScreen, null, React.createElement(Box, null, React.createElement(Text, null, 'hi'))),
      {
        exitOnCtrlC: false,
        patchConsole: false,
        stdin: new TestStdin() as NodeJS.ReadStream,
        stdout: process.stdout
      }
    )

  it('clears the alt screen without clearing host scrollback', () => {
    const instance = renderAltScreen()

    const mountWrites = stdoutWrite.mock.calls.map(([chunk]) => String(chunk)).join('')

    expect(mountWrites).toContain(ENTER_ALT_SCREEN)
    expect(mountWrites).toContain(ERASE_SCREEN)
    expect(mountWrites).toContain(CURSOR_HOME)
    expect(mountWrites).not.toContain(ERASE_SCROLLBACK)

    instance.unmount()
  })

  it('exits alt screen through React cleanup during normal unmount', () => {
    const instance = renderAltScreen()

    stdoutWrite.mockClear()
    vi.mocked(writeSync).mockClear()

    instance.unmount()

    const syncWrites = vi
      .mocked(writeSync)
      .mock.calls.map(([, chunk]) => String(chunk))
      .join('')

    const streamWrites = stdoutWrite.mock.calls.map(([chunk]) => String(chunk)).join('')

    expect(syncWrites).not.toContain(EXIT_ALT_SCREEN)
    expect(count(streamWrites, EXIT_ALT_SCREEN)).toBe(1)
  })

  it('does not double-exit alt screen during process-exit cleanup', () => {
    renderAltScreen()

    const ink = instances.get(process.stdout)

    stdoutWrite.mockClear()
    vi.mocked(writeSync).mockClear()

    ink?.unmount(0)

    const syncWrites = vi
      .mocked(writeSync)
      .mock.calls.map(([, chunk]) => String(chunk))
      .join('')

    const streamWrites = stdoutWrite.mock.calls.map(([chunk]) => String(chunk)).join('')

    expect(count(syncWrites, EXIT_ALT_SCREEN)).toBe(1)
    expect(streamWrites).not.toContain(EXIT_ALT_SCREEN)
  })
})
