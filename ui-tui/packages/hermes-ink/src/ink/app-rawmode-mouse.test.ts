import { EventEmitter } from 'events'

import React, { useContext, useEffect } from 'react'
import { describe, expect, it } from 'vitest'

import StdinContext from './components/StdinContext.js'
import Text from './components/Text.js'
import Ink from './ink.js'
import { DISABLE_MOUSE_TRACKING } from './termio/dec.js'

class FakeTty extends EventEmitter {
  chunks: string[] = []
  input: string[] = []
  columns = 80
  rows = 24
  isTTY = true
  isRaw = false

  ref(): void {}
  unref(): void {}
  read(): string | null {
    return this.input.shift() ?? null
  }
  setEncoding(): this {
    return this
  }
  setRawMode(mode: boolean): this {
    this.isRaw = mode

    return this
  }
  write(chunk: string | Uint8Array, cb?: (err?: Error | null) => void): boolean {
    this.chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
    cb?.()

    return true
  }
}

const tick = () => new Promise<void>(resolve => setImmediate(resolve))

// A child that grabs the last useInput consumer's raw-mode toggle. Mounting
// enables raw mode (count 0→1); unmounting disables it (count 1→0), which is
// the teardown path that must DISABLE_MOUSE_TRACKING so DEC 1003 hover can't
// leak as cooked-mode `35;col;row M` text over the prompt.
function RawModeConsumer({ active }: { active: boolean }) {
  const { setRawMode, isRawModeSupported } = useContext(StdinContext)

  useEffect(() => {
    if (!active || !isRawModeSupported) {
      return
    }

    setRawMode(true)

    return () => setRawMode(false)
  }, [active, isRawModeSupported, setRawMode])

  return React.createElement(Text, null, 'x')
}

describe('terminal response dispatch', () => {
  it('delivers a fragmented OSC response to the mounted input consumer', async () => {
    const stdout = new FakeTty()
    const stdin = new FakeTty()
    const responses: unknown[] = []

    function Consumer() {
      const { inputEmitter, setRawMode } = useContext(StdinContext)
      useEffect(() => {
        const onResponse = (response: unknown) => responses.push(response)
        inputEmitter.on('terminalresponse', onResponse)
        setRawMode(true)

        return () => { inputEmitter.off('terminalresponse', onResponse); setRawMode(false) }
      }, [inputEmitter, setRawMode])

      return React.createElement(Text, null, 'x')
    }

    const ink = new Ink({ exitOnCtrlC: false, patchConsole: false,
      stderr: new FakeTty() as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream, stdout: stdout as unknown as NodeJS.WriteStream })

    ink.render(React.createElement(Consumer))
    ink.onRender()
    await tick()
    stdin.input.push('\x1b]777;', 'stop;session;4\x07')
    stdin.emit('readable')
    expect(responses).toEqual([{ type: 'osc', code: 777, data: 'stop;session;4' }])
    ink.unmount()
  })
})

describe('App raw-mode teardown', () => {
  it('disables mouse tracking when the last raw-mode consumer detaches', async () => {
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

    // Mouse tracking is asserted on the alt screen; the teardown path lives in
    // App, independent of who enabled tracking.
    ink.setAltScreenActive(true, 'all')
    ink.render(React.createElement(RawModeConsumer, { active: true }))
    ink.onRender()
    await tick()
    expect(stdin.isRaw).toBe(true)

    stdout.chunks = []

    // Drop the consumer → raw-mode count hits 0 → teardown runs.
    ink.render(React.createElement(RawModeConsumer, { active: false }))
    ink.onRender()
    await tick()

    expect(stdin.isRaw).toBe(false)
    expect(stdout.chunks.join('')).toContain(DISABLE_MOUSE_TRACKING)

    ink.unmount()
  })
})
