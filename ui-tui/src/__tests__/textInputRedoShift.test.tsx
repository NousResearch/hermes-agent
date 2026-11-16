import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { TextInput } from '../components/textInput.js'
import type * as PlatformModule from '../lib/platform.js'

// Treat the kitty `super` bit as the action modifier regardless of host OS so
// the Cmd+Shift+Z chord below is exercised on every CI lane.
vi.mock('../lib/platform.js', async importOriginal => {
  const mod = await importOriginal<typeof PlatformModule>()

  return {
    ...mod,
    isActionMod: (key: { ctrl: boolean; meta: boolean; super?: boolean }) => key.ctrl || key.super === true
  }
})

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0

  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }

  ref = vi.fn()

  send(...chunks: string[]) {
    this.chunks.push(...chunks)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }

  setEncoding = vi.fn()

  setRawMode = vi.fn((enabled: boolean) => {
    this.isRaw = enabled
  })

  unref = vi.fn()
}

const settle = (ms = 25) => new Promise(resolve => setTimeout(resolve, ms))

function makeStreams() {
  const stdin = new FakeInput()
  const stdout = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stderr, { columns: 80, isTTY: false, rows: 24 })

  return { stderr, stdin, stdout }
}

describe('TextInput redo chord under extended-key terminals', () => {
  it('Cmd+Shift+Z delivered as uppercase "Z" (kitty CSI-u) redoes instead of typing Z', async () => {
    const streams = makeStreams()
    const changes: string[] = []

    function Harness() {
      const [value, setValue] = useState('')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onSubmit={() => {}}
          value={value}
        />
      )
    }

    const instance = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: streams.stderr as NodeJS.WriteStream,
      stdin: streams.stdin as unknown as NodeJS.ReadStream,
      stdout: streams.stdout as NodeJS.WriteStream
    })

    await settle()

    streams.stdin.send('a')
    await settle()
    streams.stdin.send('b')
    await settle()
    expect(changes.at(-1)).toBe('ab')

    // Undo: super+z (CSI-u, modifier 9 = 1 + super bit 8).
    streams.stdin.send('\u001b[122;9u')
    await settle()
    expect(changes.at(-1)).toBe('a')

    // Redo: super+shift+z (modifier 10). hermes-ink restores the shifted
    // letter's case, so the composer sees inp 'Z' with key.shift set.
    streams.stdin.send('\u001b[122;10u')
    await settle()

    instance.unmount()
    instance.cleanup()

    expect(changes.at(-1)).toBe('ab')
    expect(changes).not.toContain('aZ')
  })
})
