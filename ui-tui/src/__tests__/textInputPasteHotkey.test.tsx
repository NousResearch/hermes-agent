import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { TextInput, type PasteEvent } from '../components/textInput.js'

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

const settle = (ms = 0) => new Promise(resolve => setTimeout(resolve, ms))

function makeStreams() {
  const stdin = new FakeInput()
  const stdout = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stderr, { columns: 80, isTTY: false, rows: 24 })

  return { stderr, stdin, stdout }
}

function Harness({
  onChange,
  onPaste
}: {
  onChange: (value: string) => void
  onPaste: (event: PasteEvent) => null
}) {
  const [value, setValue] = useState('')

  return (
    <TextInput
      columns={80}
      onChange={next => {
        onChange(next)
        setValue(next)
      }}
      onPaste={onPaste}
      value={value}
    />
  )
}

async function triggerPasteHotkey(sequence: string) {
  const streams = makeStreams()
  const values: string[] = []
  const onPaste = vi.fn((event: PasteEvent) => {
    expect(event.hotkey).toBe(true)
    expect(event.text).toBe('')
    expect(event.value).toBe('')
    expect(event.cursor).toBe(0)

    return null
  })

  const instance = renderSync(<Harness onChange={value => values.push(value)} onPaste={onPaste} />, {
    patchConsole: false,
    stderr: streams.stderr as NodeJS.WriteStream,
    stdin: streams.stdin as unknown as NodeJS.ReadStream,
    stdout: streams.stdout as NodeJS.WriteStream
  })

  try {
    await settle()
    streams.stdin.send(sequence)
    await settle(25)
  } finally {
    instance.unmount()
    instance.cleanup()
  }

  return { onPaste, values }
}

describe('TextInput paste hotkeys', () => {
  it.each([
    ['raw ctrl+v / ctrl+shift+v', '\x16'],
    ['alt+v', '\x1bv'],
    ['kitty csi-u ctrl+v', '\x1b[118;5u'],
    ['kitty csi-u ctrl+shift+v', '\x1b[118;6u'],
    ['modifyOtherKeys ctrl+v', '\x1b[27;5;118~'],
    ['modifyOtherKeys ctrl+shift+v', '\x1b[27;6;118~']
  ])('routes %s through onPaste instead of typing a literal v', async (_label, sequence) => {
    const { onPaste, values } = await triggerPasteHotkey(sequence)

    expect(onPaste).toHaveBeenCalledTimes(1)
    expect(values).toEqual([])
  })
})
