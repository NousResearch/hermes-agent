import { PassThrough } from 'stream'

import { renderSync, Text } from '@hermes/ink'
import React, { useMemo, useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'

const ref = <T,>(current: T) => ({ current })
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

describe('voice.submit_mode live composer rendering', () => {
  it('renders a draft transcript in the real React composer state without submitting it', async () => {
    const exposed = { current: null as null | ((event: any) => void) }
    const submit = vi.fn()

    function Harness() {
      const [input, setInput] = useState('existing draft')

      const handler = useMemo(
        () =>
          createGatewayEventHandler({
            composer: {
              dequeue: () => undefined,
              queueEditRef: ref<null | number>(null),
              sendQueued: vi.fn(),
              setInput
            },
            gateway: {
              gw: { request: vi.fn() },
              rpc: vi.fn(async (method: string) =>
                method === 'config.get' ? { config: { voice: { submit_mode: 'draft' } } } : null
              )
            },
            session: {
              STARTUP_RESUME_ID: '',
              colsRef: ref(80),
              newSession: vi.fn(),
              resetSession: vi.fn(),
              resumeById: vi.fn(),
              setCatalog: vi.fn()
            },
            submission: { submitRef: ref(submit) },
            system: { bellOnComplete: false, sys: vi.fn() },
            transcript: {
              appendMessage: vi.fn(),
              panel: vi.fn(),
              setHistoryItems: vi.fn()
            },
            voice: {
              setProcessing: vi.fn(),
              setRecording: vi.fn(),
              setVoiceEnabled: vi.fn()
            }
          } as any),
        []
      )

      exposed.current = handler

      return <Text>{input || 'empty composer'}</Text>
    }

    const stdin = new PassThrough()
    const stdout = new PassThrough()
    const stderr = new PassThrough()
    let output = ''

    Object.assign(stdin, { isTTY: false })
    Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const instance = renderSync(<Harness />, {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    })

    try {
      expect(exposed.current).not.toBeNull()
      exposed.current!({ payload: { text: 'editable voice draft' }, type: 'voice.transcript' })

      for (let i = 0; i < 20 && !output.includes('editable voice draft'); i += 1) {
        await delay(10)
      }

      expect(output).toContain('existing draft editable voice draft')
      expect(submit).not.toHaveBeenCalled()
    } finally {
      instance.unmount()
    }
  })
})
