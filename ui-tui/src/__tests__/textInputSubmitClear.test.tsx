import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { TextInput } from '../components/textInput.js'

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

describe('TextInput submit clearing', () => {
  it('accepts the parent clear after a Korean IME commit immediately followed by Enter', async () => {
    const streams = makeStreams()
    const changes: string[] = []
    const submits: string[] = []

    function Harness() {
      const [value, setValue] = useState('')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onSubmit={text => {
            submits.push(text)
            setValue('')
          }}
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

    const prefix = '한글을 사용하면 마지막 문자가 남아있는 버그가 있어 리포트해'
    const finalSyllable = '줘'
    const full = prefix + finalSyllable

    streams.stdin.send(prefix)
    await settle(25)

    streams.stdin.send(finalSyllable, '\r')
    await settle(25)

    streams.stdin.send('x')
    await settle(25)

    instance.unmount()
    instance.cleanup()

    expect(submits).toEqual([full])
    expect(changes.at(-1)).toBe('x')
    expect(changes).not.toContain(`${full}x`)
  })

  it('clears the composer when text and a CSI-u Enter arrive in one read', async () => {
    const streams = makeStreams()
    const changes: string[] = []
    const submits: string[] = []

    function Harness() {
      const [value, setValue] = useState('')

      return (
        <TextInput
          columns={80}
          onChange={next => {
            changes.push(next)
            setValue(next)
          }}
          onSubmit={text => {
            submits.push(text)
            setValue('')
          }}
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

    // Text and the Enter key share one read and one React batch, so the parent value ends
    // where it started ('') and the draft only clears if TextInput re-syncs on submit.
    streams.stdin.send('hello\x1b[13u')
    await settle(25)

    streams.stdin.send('x')
    await settle(25)

    instance.unmount()
    instance.cleanup()

    expect(submits).toEqual(['hello'])
    expect(changes.at(-1)).toBe('x')
    expect(changes).not.toContain('hellox')
  })

  describe('keys after a CSI-u Enter in the same read', () => {
    afterEach(() => {
      vi.unstubAllEnvs()
      vi.restoreAllMocks()
      ;(process.stdout as { isTTY?: boolean }).isTTY = undefined
    })

    // One read carries text, a kitty Enter and the first key of the next draft, so all three
    // land in one React batch before the parent's submit response renders.
    async function burst(clearOnSubmit: boolean, fastEcho: boolean, read = 'hello\x1b[13ux') {
      const echoed: string[] = []

      if (fastEcho) {
        // useStdout() resolves to process.stdout, so the fast-echo bypass is armed there.
        vi.stubEnv('TERM_PROGRAM', 'iTerm.app')
        vi.stubEnv('TMUX', '')
        ;(process.stdout as { isTTY?: boolean }).isTTY = true
        vi.spyOn(process.stdout, 'write').mockImplementation(chunk => {
          echoed.push(String(chunk))

          return true
        })
      }

      const streams = makeStreams()
      const changes: string[] = []
      const submits: string[] = []
      let current = ''

      function Harness() {
        const [value, setValue] = useState('')
        current = value

        return (
          <TextInput
            columns={80}
            onChange={next => {
              changes.push(next)
              setValue(next)
            }}
            onSubmit={text => {
              submits.push(text)

              if (clearOnSubmit) {
                setValue('')
              }
            }}
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
      streams.stdin.send(read)
      await settle(50)
      streams.stdin.send('y')
      await settle(50)

      instance.unmount()
      instance.cleanup()

      return { changes, current, echoed: echoed.join(''), submits }
    }

    for (const fastEcho of [true, false]) {
      const mode = fastEcho ? 'fast echo' : 'Ink render'

      it(`starts the cleared draft with the key typed after Enter (${mode})`, async () => {
        const { changes, current, echoed, submits } = await burst(true, fastEcho)

        expect(submits).toEqual(['hello'])
        expect(current).toBe('xy')
        expect(changes.at(-1)).toBe('xy')
        expect(changes).not.toContain('hellox')

        if (fastEcho) {
          // The bypass is really armed: the next draft's later keystroke is written directly.
          expect(echoed).toContain('y')
        }
      })

      it(`keeps the draft and the key typed after Enter when the parent does not clear (${mode})`, async () => {
        const { current, submits } = await burst(false, fastEcho)

        expect(submits).toEqual(['hello'])
        expect(current).toBe('helloxy')
      })
    }

    it('submits each line and keeps the tail when one read holds two CSI-u Enters', async () => {
      const { current, submits } = await burst(true, true, 'a\x1b[13ub\x1b[13uc')

      expect(submits).toEqual(['a', 'b'])
      expect(current).toBe('cy')
    })
  })
})
