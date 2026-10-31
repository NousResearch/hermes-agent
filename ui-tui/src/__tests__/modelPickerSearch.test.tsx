import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import { stripAnsi } from '@hermes/shared/ansi'
import React from 'react'
import { expect, it, vi } from 'vitest'

const inputHarness = vi.hoisted(() => ({
  handler: undefined as undefined | ((input: string, key: Record<string, boolean>) => void)
}))

// Stub useInput (as subscriptionOverlay.test does) and drive the picker's own handler, registered last.
vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal()

  return {
    ...mod,
    useInput: (handler: (input: string, key: Record<string, boolean>) => void) => {
      inputHarness.handler = handler
    }
  }
})

import { ModelPicker, modelPickerCommand } from '../components/modelPicker.js'
import type { GatewayClient } from '../gatewayClient.js'
import { DEFAULT_THEME } from '../theme.js'

it('switches to a typed model on the provider that serves it', async () => {
  const providers = [
    { name: 'OpenRouter', slug: 'openrouter', models: ['qwen/qwen3-max'] },
    { name: 'Alibaba', slug: 'alibaba', models: ['qwen3-max', 'qwen3-max-mini'] }
  ]

  const request = vi.fn(async () => ({ model: 'qwen3-max', providers }))
  const onCancel = vi.fn()
  const onSelect = vi.fn()
  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 40 })
  let output = ''

  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <ModelPicker
      gw={{ request } as unknown as GatewayClient}
      onCancel={onCancel}
      onSelect={onSelect}
      sessionId="s1"
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      stdin: Object.assign(new PassThrough(), { isTTY: false }) as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    }
  )

  try {
    await vi.waitFor(() => expect(stripAnsi(output)).toContain('OpenRouter'), { timeout: 5000 })

    // A leading 'q' is search text, not the close key.
    for (const ch of 'qwen3-max-mini') {
      inputHarness.handler?.(ch, {})
    }

    await vi.waitFor(() => expect(stripAnsi(output)).toContain('Alibaba · qwen3-max-mini'), { timeout: 5000 })
    inputHarness.handler?.('', { return: true })

    expect(onCancel).not.toHaveBeenCalled()
    expect(onSelect).toHaveBeenCalledWith(modelPickerCommand('qwen3-max-mini', 'alibaba', false))
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})


it.each([
  {
    name: 'a rejected model.options request',
    request: async () => {
      throw new Error('gateway offline')
    },
    terminalText: 'error: gateway offline'
  },
  {
    name: 'an empty provider response',
    request: async () => ({ model: '', providers: [] }),
    terminalText: 'no providers available'
  }
])('cancels immediately from $name without hidden filter state', async ({ request: requestImpl, terminalText }) => {
  const request = vi.fn(requestImpl)
  const onCancel = vi.fn()
  const onSelect = vi.fn()
  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 40 })
  let output = ''

  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <ModelPicker
      gw={{ request } as unknown as GatewayClient}
      onCancel={onCancel}
      onSelect={onSelect}
      sessionId="s1"
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      stdin: Object.assign(new PassThrough(), { isTTY: false }) as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream
    }
  )

  try {
    await vi.waitFor(() => expect(stripAnsi(output)).toContain(terminalText), { timeout: 5000 })

    // q is the advertised close key on terminal views, not search text.
    inputHarness.handler?.('q', {})
    expect(onCancel).toHaveBeenCalledTimes(1)

    onCancel.mockClear()

    // Other printable keys must not create an invisible filter that swallows Esc.
    inputHarness.handler?.('x', {})
    expect(onCancel).not.toHaveBeenCalled()
    inputHarness.handler?.('', { escape: true })
    expect(onCancel).toHaveBeenCalledTimes(1)
    expect(onSelect).not.toHaveBeenCalled()
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})
