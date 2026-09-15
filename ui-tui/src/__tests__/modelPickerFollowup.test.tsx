import { PassThrough } from 'stream'
import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

const input = vi.hoisted(() => ({
  stdout: null as unknown as NodeJS.WriteStream,
  handlers: [] as Array<(ch: string, key: Record<string, boolean>) => void>
}))
vi.mock('@hermes/ink', async importOriginal => ({
  ...(await importOriginal()),
  useStdout: () => ({ stdout: input.stdout }),
  useInput: (handler: (ch: string, key: Record<string, boolean>) => void) => {
    input.handlers.push(handler)
  }
}))
import { ModelPicker } from '../components/modelPicker.js'
import type { GatewayClient } from '../gatewayClient.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

it.each([
  [100, 40],
  [50, 18],
  [46, 16]
])('keeps warning and reset across both stages at %sx%s', async (columns, rows) => {
  const warning =
    'Credential pool has a recorded limit; remote availability and model scope not checked. Recorded reset: 2100-01-01T00:00:00+00:00.'
  const gw = {
    request: vi.fn().mockResolvedValue({
      providers: [
        {
          slug: 'openai-codex',
          name: 'OpenAI Codex',
          authenticated: true,
          warning,
          models: Array.from({ length: 24 }, (_, i) => `gpt-synthetic-${i}`)
        }
      ]
    })
  } as unknown as GatewayClient
  const stdout = new PassThrough(),
    stdin = new PassThrough(),
    stderr = new PassThrough()
  Object.assign(stdout, { columns, rows, isTTY: false })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  let frame = ''
  stdout.on('data', chunk => {
    const text = stripAnsi(chunk.toString())
    if (text.trim()) frame = text
  })
  input.handlers = []
  input.stdout = stdout as unknown as NodeJS.WriteStream
  const instance = renderSync(
    React.createElement(ModelPicker, { gw, sessionId: null, t: DEFAULT_THEME, onCancel: () => {}, onSelect: () => {} }),
    {
      patchConsole: false,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: stderr as unknown as NodeJS.WriteStream
    }
  )
  const verify = (stage: string) => {
    expect(frame).toContain(stage)
    expect(frame.replace(/\s+/g, ' ')).toContain('not checked.')
    expect(frame).toContain('2100-01-01T00:00:00+00:00.')
    expect(frame.trimEnd().split('\n').length).toBeLessThanOrEqual(rows)
  }
  try {
    await vi.waitFor(() => verify('Select provider'))
    // The last useInput is the real ModelPicker callback from the latest render.
    input.handlers.at(-1)!('', { return: true })
    await vi.waitFor(() => verify('Select model'))
    expect(frame).toContain('gpt-synthetic-0')
    for (let i = 0; i < 20; i++) {
      input.handlers.at(-1)!('', { downArrow: true })
      await new Promise(resolve => setTimeout(resolve, 20))
    }
    await vi.waitFor(() => {
      verify('Select model')
      expect(frame).toContain('gpt-synthetic-20')
    })
    expect(gw.request).toHaveBeenCalledTimes(1)
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})
