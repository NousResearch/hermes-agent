import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { GatewayClient } from '../gatewayClient.js'
import { TriggerPicker } from '../components/triggerPicker.js'
import { DEFAULT_THEME } from '../theme.js'

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

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 24 })
  Object.assign(stderr, { columns: 100, isTTY: false, rows: 24 })

  return { stderr, stdin, stdout }
}

const makeGw = (triggers: Array<{ phrase: string; description: string }>) =>
  ({
    request: vi.fn((method: string) => {
      if (method === 'trigger.list') {
        return Promise.resolve({ triggers })
      }
      return Promise.resolve(null)
    })
  }) as unknown as GatewayClient

const mountPicker = ({ triggers, onPick = vi.fn(), onClose = vi.fn() }: { triggers: Array<{ phrase: string; description: string }>; onPick?: (phrase: string) => void; onClose?: () => void }) => {
  const streams = makeStreams()
  const gw = makeGw(triggers)

  const instance = renderSync(
    <TriggerPicker gw={gw} onClose={onClose} onPick={onPick} t={DEFAULT_THEME} />,
    {
      patchConsole: false,
      stderr: streams.stderr as unknown as NodeJS.WriteStream,
      stdin: streams.stdin as unknown as NodeJS.ReadStream,
      stdout: streams.stdout as unknown as NodeJS.WriteStream
    }
  )

  return { ...streams, gw, instance, onPick, onClose }
}

describe('TriggerPicker', () => {
  it('requests the trigger list on mount', async () => {
    const { gw, instance } = mountPicker({
      triggers: [{ phrase: 'todo tracking', description: 'Track a multi-step objective' }]
    })
    await settle(20)
    expect(gw.request).toHaveBeenCalledWith('trigger.list')
    instance.unmount()
  })

  it('renders the loaded triggers', async () => {
    const { instance, stdout } = mountPicker({
      triggers: [
        { phrase: 'todo tracking', description: 'Track a multi-step objective' },
        { phrase: 'gate notebook', description: 'Investigate a gate notebook' }
      ]
    })
    let out = ''
    stdout.on('data', (chunk: Buffer) => {
      out += chunk.toString()
    })
    await settle(30)
    expect(out).toContain('todo tracking')
    expect(out).toContain('gate notebook')
    instance.unmount()
  })

  it('dispatches the highlighted phrase on Enter', async () => {
    const { instance, stdin, onPick } = mountPicker({
      triggers: [{ phrase: 'todo tracking', description: 'Track a multi-step objective' }]
    })
    await settle(20)
    stdin.send('\r')
    await settle(20)
    expect(onPick).toHaveBeenCalledWith('todo tracking')
    instance.unmount()
  })

  it('closes on Escape', async () => {
    const onClose = vi.fn()
    const { instance, stdin, onPick } = mountPicker({
      triggers: [{ phrase: 'todo tracking', description: 'Track a multi-step objective' }],
      onPick: vi.fn()
    })
    await settle(20)
    stdin.send('\u001b')
    await settle(20)
    // Escape closes the picker without dispatching.
    expect(onPick).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    instance.unmount()
  })
})