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
    { name: 'OpenRouter', slug: 'openrouter', models: ['openai/gpt-6'] },
    { name: 'OpenAI', slug: 'openai', models: ['gpt-6', 'gpt-6-mini'] }
  ]

  const request = vi.fn(async () => ({ model: 'gpt-6', providers }))
  const onSelect = vi.fn()
  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 40 })
  let output = ''

  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <ModelPicker
      gw={{ request } as unknown as GatewayClient}
      onCancel={() => {}}
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

    for (const ch of 'gpt-6-mini') {
      inputHarness.handler?.(ch, {})
    }

    await vi.waitFor(() => expect(stripAnsi(output)).toContain('OpenAI · gpt-6-mini'), { timeout: 5000 })
    inputHarness.handler?.('', { return: true })

    expect(onSelect).toHaveBeenCalledWith(modelPickerCommand('gpt-6-mini', 'openai', false))
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})
