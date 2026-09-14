import { PassThrough } from 'stream'
import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

vi.mock('@hermes/ink', async importOriginal => ({
  ...(await importOriginal()),
  useInput: () => {}
}))

import { ModelPicker } from '../components/modelPicker.js'
import type { GatewayClient } from '../gatewayClient.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

it('renders the persisted limit and reset without hiding the provider', async () => {
  const warning =
    'Credential pool has a recorded limit; remote availability and model scope not checked. Recorded reset: 2100-01-01T00:00:00+00:00.'
  const gw = {
    request: vi.fn().mockResolvedValue({
      providers: [
        {
          slug: 'openai-codex',
          name: 'OpenAI Codex',
          models: ['gpt-5.3-codex'],
          authenticated: true,
          warning
        }
      ]
    })
  } as unknown as GatewayClient
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  Object.assign(stdout, { columns: 100, rows: 40, isTTY: false })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })
  const instance = renderSync(
    React.createElement(ModelPicker, {
      gw,
      t: DEFAULT_THEME,
      onCancel: () => {},
      onSelect: () => {}
    }),
    {
      patchConsole: false,
      stdout: stdout as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stderr: stderr as NodeJS.WriteStream
    }
  )
  try {
    await vi.waitFor(() => {
      expect(output).toContain('OpenAI Codex')
      expect(output).toContain('recorded limit')
      expect(output).toContain('2100-01-01')
    })
    expect(gw.request).toHaveBeenCalledTimes(1)
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})
