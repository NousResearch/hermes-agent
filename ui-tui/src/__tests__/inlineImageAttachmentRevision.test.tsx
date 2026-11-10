import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import type { ImageRenderResult } from '@hermes/shared/gateway-events'
import { afterEach, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { MessageLine } from '../components/messageLine.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const unmounts: Array<() => void> = []

afterEach(() => {
  unmounts.splice(0).forEach(fn => fn())
  vi.restoreAllMocks()
})

const settle = async () => {
  for (let i = 0; i < 6; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
}

// The user turn exactly as submitPrompt appends it for a send carrying `attachments`.
const sentTurn = (text: string, path: string): Msg => {
  resetUiState()
  patchUiState({ sid: 'sess' })

  const appendMessage = vi.fn()
  const gw = { request: vi.fn(async () => ({ status: 'streaming' })) } as unknown as GatewayClient

  const deps: SubmitPromptDeps = {
    appendMessage,
    enqueue: vi.fn(),
    expand: value => value,
    gw,
    setLastUserMsg: vi.fn(),
    sys: vi.fn()
  }

  submitPrompt(text, deps, true, undefined, { attachments: [{ mime: 'image/png', path }], skipDetectDrop: true })

  return appendMessage.mock.calls[0]![0] as Msg
}

const paint = (msg: Msg, render: ImageRenderResult) => {
  const request = vi.fn(async () => render)
  const gateway = { gw: { request } as unknown as GatewayClient, rpc: vi.fn() } as unknown as GatewayServices
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 60 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <GatewayProvider value={gateway}>
      <MessageLine cols={100} msg={msg} t={DEFAULT_THEME} />
    </GatewayProvider>,
    {
      patchConsole: false,
      stderr: stderr as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    }
  )

  unmounts.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return { output: () => output, request }
}

it('re-attaching a file overwritten at the same path shows the new picture', async () => {
  // The legacy attach path keeps the user's own path, so both turns name the same file.
  const path = '/tmp/overwritten-attachment.png'
  const px = [255, 0, 0, 255, 255, 0, 0, 255]
  const first = paint(sentTurn('look', path), { cells: [[px]], cols: 1, mode: 'unicode', rows: 1 })

  await settle()
  expect(first.output()).not.toContain('▀▀▀')

  // The file was overwritten with a wider picture before the second send.
  const second = paint(sentTurn('and now?', path), { cells: [[px, px, px]], cols: 3, mode: 'unicode', rows: 1 })

  await settle()
  expect(second.request).toHaveBeenCalledTimes(1)
  expect(second.output()).toContain('▀▀▀')
})
