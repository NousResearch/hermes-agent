import { EventEmitter } from 'events'

import { describe, expect, it } from 'vitest'

import Ink from './ink.js'
import { CURSOR_HOME, ERASE_SCREEN } from './termio/csi.js'
import {
  BSU,
  DISABLE_MOUSE_TRACKING,
  enableMouseTrackingFor,
  ENTER_ALT_SCREEN,
  ESU,
  EXIT_ALT_SCREEN
} from './termio/dec.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 80
  rows = 24
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const makeInk = () => {
  const stdout = new FakeTty()
  const stdin = new FakeTty()
  const stderr = new FakeTty()

  const ink = new Ink({
    exitOnCtrlC: false,
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return { ink, stdout }
}

describe('Ink hard alt-screen reset', () => {
  it('writes an exact leave-enter reset and restores synchronized output and mouse state', () => {
    const { ink, stdout } = makeInk()

    ink.setAltScreenActive(true, 'wheel')
    ink.hardResetScreen()

    const reset = stdout.chunks[0]

    const expected =
      ESU +
      DISABLE_MOUSE_TRACKING +
      EXIT_ALT_SCREEN +
      ENTER_ALT_SCREEN +
      BSU +
      ERASE_SCREEN +
      CURSOR_HOME +
      DISABLE_MOUSE_TRACKING +
      enableMouseTrackingFor('wheel') +
      ESU

    expect(reset).toBe(expected)
    expect(reset.indexOf('\x1b[?1049l')).toBeGreaterThanOrEqual(0)
    expect(reset.indexOf('\x1b[?1049l')).toBeLessThan(reset.indexOf('\x1b[?1049h'))

    ink.unmount()
  })
})
