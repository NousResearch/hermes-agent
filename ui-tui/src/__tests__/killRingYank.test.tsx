import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useCallback, useState } from 'react'
import { expect, it, vi } from 'vitest'

import { TextInput } from '../components/textInput.js'

/**
 * Drive the real composer through stdin. The kill ring only proves itself end to end: a chord
 * removes text, `Alt+Y` brings it back, `Ctrl+Z` undoes the yank.
 */
const mountComposer = (initial = '') => {
  const stdin = Object.assign(new PassThrough(), {
    isTTY: true,
    ref: vi.fn(),
    setRawMode: vi.fn(),
    unref: vi.fn()
  })

  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 })
  const seen: string[] = []

  const Harness = () => {
    const [value, setValue] = useState(initial)

    const onChange = useCallback((next: string) => {
      seen.push(next)
      setValue(next)
    }, [])

    return <TextInput columns={80} onChange={onChange} onSubmit={vi.fn()} value={value} />
  }

  const instance = renderSync(<Harness />, {
    patchConsole: false,
    stderr: new PassThrough() as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  const settle = () => new Promise(resolve => setTimeout(resolve, 30))

  return {
    current: () => seen.at(-1) ?? initial,
    key: async (seq: string) => {
      stdin.write(seq)
      await settle()
    },
    type: async (text: string) => {
      for (const ch of text) {
        stdin.write(ch)
      }

      await settle()
    },
    unmount: () => instance.unmount()
  }
}

it('kills a word with Ctrl+W and brings it back with Alt+Y', async () => {
  const c = mountComposer()

  try {
    await c.type('hello world')
    expect(c.current()).toBe('hello world')

    await c.key('\u0017')
    expect(c.current()).toBe('hello ')

    await c.key('\u001by')
    expect(c.current()).toBe('hello world')
  } finally {
    c.unmount()
  }
})

it('cycles to older cuts on a second Alt+Y and wraps at the end of the ring', async () => {
  const c = mountComposer()

  try {
    await c.type('alpha beta')
    await c.key('\u0017') // kills "beta"
    await c.type('gamma delta')
    await c.key('\u0017') // kills "delta"
    expect(c.current()).toBe('alpha gamma ')

    await c.key('\u001by')
    expect(c.current()).toBe('alpha gamma delta')

    await c.key('\u001by')
    expect(c.current()).toBe('alpha gamma beta')

    await c.key('\u001by') // wraps back to the newest cut
    expect(c.current()).toBe('alpha gamma delta')
  } finally {
    c.unmount()
  }
})

it('undoes a yank, as readline does', async () => {
  const c = mountComposer()

  try {
    await c.type('one two')
    await c.key('\u0017')
    expect(c.current()).toBe('one ')

    await c.key('\u001by')
    expect(c.current()).toBe('one two')

    await c.key('\u001a')
    expect(c.current()).toBe('one ')
  } finally {
    c.unmount()
  }
})

it('leaves Ctrl+Y as redo and an empty ring as a no-op', async () => {
  const c = mountComposer()

  try {
    await c.key('\u001by') // nothing has been cut
    expect(c.current()).toBe('')

    await c.type('ab')
    await c.key('\u0017')
    expect(c.current()).toBe('')
    await c.key('\u001by')
    expect(c.current()).toBe('ab')

    await c.key('\u001a') // undo the yank
    expect(c.current()).toBe('')

    // Ctrl+Y is untouched: it still redoes what the undo just took back.
    await c.key('\u0019')
    expect(c.current()).toBe('ab')
  } finally {
    c.unmount()
  }
})
