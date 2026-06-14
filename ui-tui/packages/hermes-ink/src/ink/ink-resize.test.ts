import { EventEmitter } from 'events'

import React from 'react'
import { describe, expect, it } from 'vitest'

import Text from './components/Text.js'
import Ink from './ink.js'
import { CURSOR_HOME, ERASE_SCREEN } from './termio/csi.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  columns = 20
  rows = 5
  isTTY = true

  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const tick = () => new Promise<void>(resolve => queueMicrotask(resolve))
const macrotask = () => new Promise<void>(resolve => setTimeout(resolve, 0))

describe('Ink resize healing', () => {
  it('heals same-dimension alt-screen resize events with an erase before repaint', async () => {
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

    ink.setAltScreenActive(true)
    ink.render(React.createElement(Text, null, 'hello'))
    ink.onRender()
    stdout.chunks = []

    stdout.emit('resize')
    await tick()
    await macrotask()
    ink.onRender()
    await tick()

    const output = stdout.chunks.join('')
    expect(output).toContain(ERASE_SCREEN)
    expect(output).toContain(CURSOR_HOME)

    ink.unmount()
  })
})
