import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { expect, it, vi } from 'vitest'

import { getInputSelection } from '../app/inputSelectionStore.js'
import { TextInput } from '../components/textInput.js'

class Input extends EventEmitter {
  chunks: string[] = []
  isTTY = true
  readableLength = 0
  read() {
    return this.chunks.shift() ?? null
  }
  ref() {}
  unref() {}
  setEncoding() {}
  setRawMode() {}
  send(text: string) {
    this.chunks.push(text)
    this.emit('readable')
  }
}

it('dispatches Up into the previous wrapped row before inserting the next character', async () => {
  const stdin = new Input()
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })
  const changes: string[] = []

  function Prompt() {
    const [value, setValue] = useState('abcdefghijklmnopqrstuvwx')

    return (
      <TextInput
        columns={12}
        onChange={next => {
          changes.push(next)
          setValue(next)
        }}
        value={value}
      />
    )
  }

  const instance = renderSync(<Prompt />, {
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stderr: stdout as unknown as NodeJS.WriteStream,
    patchConsole: false
  })

  try {
    await vi.waitFor(() => expect(getInputSelection()?.start).toBe(24))
    stdin.send('\u001b[A')
    await vi.waitFor(() => expect(getInputSelection()?.start).toBe(11))
    stdin.send('X')
    await vi.waitFor(() => expect(changes.at(-1)).toBe('abcdefghijkXlmnopqrstuvwx'))
  } finally {
    instance.unmount()
  }
})
