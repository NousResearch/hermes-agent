import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useRef, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useInputHandlers } from '../app/useInputHandlers.js'
import { TextInput } from '../components/textInput.js'
import { isMac } from '../lib/platform.js'

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

// The host's action chord for K: Cmd+K (kitty keyboard protocol, super) on macOS, Ctrl+K elsewhere.
const ACTION_K = isMac ? '\x1b[107;9u' : '\x0b'

afterEach(() => {
  resetUiState()
  resetOverlayState()
})

describe('kill-to-end with messages queued', () => {
  it('kills the rest of the draft line without sending the next queued message', async () => {
    patchUiState({ busy: true, sid: 'sid-1' })

    const stdin = new FakeInput()
    const stdout = new PassThrough()
    const stderr = new PassThrough()

    Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
    Object.assign(stderr, { columns: 80, isTTY: false, rows: 24 })

    const dispatched: string[] = []
    const values: string[] = []
    const queue = [{ display: 'queued follow-up', text: 'queued follow-up' }]

    const noop = () => {}

    function Harness() {
      const [input, setInput] = useState('fix the parser please')
      const queueRef = useRef(queue)

      useInputHandlers({
        actions: {
          answerClarify: noop,
          appendMessage: noop,
          die: noop,
          dispatchSubmission: full => void dispatched.push(full),
          guardBusySessionSwitch: () => false,
          newSession: noop,
          sys: noop
        },
        composer: {
          actions: {
            clearIn: noop,
            dequeue: () => queueRef.current.shift()?.text,
            setCompIdx: noop,
            setHistoryIdx: noop,
            setInput,
            setQueueEdit: noop
          } as never,
          refs: {
            historyDraftRef: { current: '' },
            historyRef: { current: [] },
            queueEditRef: { current: null },
            queueRef
          } as never,
          state: {
            compIdx: 0,
            compReplace: 0,
            completions: [],
            historyIdx: null,
            input,
            inputBuf: [],
            queueEditIdx: null,
            queuedDisplay: queueRef.current.map(item => item.display),
            tokens: []
          }
        },
        gateway: { gw: {} as never, rpc: (async () => null) as never },
        terminal: {
          hasSelection: false,
          scrollRef: { current: null },
          scrollWithSelection: noop,
          selection: { clearSelection: noop } as never
        },
        voice: {
          enabled: false,
          recordKey: {} as never,
          recording: false,
          setProcessing: noop,
          setRecording: noop,
          setVoiceEnabled: noop,
          setVoiceTts: noop
        },
        wheelStep: 1
      })

      return (
        <TextInput
          columns={80}
          onChange={next => {
            values.push(next)
            setInput(next)
          }}
          onSubmit={noop}
          value={input}
        />
      )
    }

    const instance = renderSync(React.createElement(Harness), {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    await settle()
    stdin.send('\x1b[D'.repeat(' please'.length))
    await settle(25)
    stdin.send(ACTION_K)
    await settle(25)

    instance.unmount()
    instance.cleanup()

    expect(values.at(-1)).toBe('fix the parser')
    expect(dispatched).toEqual([])
    expect(queue.map(item => item.text)).toEqual(['queued follow-up'])
  })
})
