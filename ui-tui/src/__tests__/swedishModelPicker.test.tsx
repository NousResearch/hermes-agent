import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

const input = vi.hoisted(() => ({
  handler: undefined as undefined | ((text: string, key: Record<string, boolean>) => void)
}))

vi.mock('@hermes/ink', async importOriginal => ({
  ...(await importOriginal()),
  useInput: (handler: (text: string, key: Record<string, boolean>) => void) => {
    input.handler = handler
  }
}))

import { ModelPicker } from '../components/modelPicker.js'
import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import { setTuiLanguage } from '../i18n/index.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

it('renders Swedish provider/model stages while preserving selected model and provider IDs', async () => {
  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 40 })
  const stdin = Object.assign(new PassThrough(), { isTTY: false })
  const stderr = Object.assign(new PassThrough(), { isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const request = vi.fn(async () => ({
    model: 'vendor/model-fixture',
    providers: [
      {
        slug: 'provider-fixture',
        name: 'Fixture Provider',
        authenticated: true,
        is_current: true,
        models: ['vendor/model-fixture']
      }
    ]
  }))

  const onSelect = vi.fn()
  setTuiLanguage('sv')

  const element = (
    <ModelPicker
      gw={{ request } as unknown as GatewayClient}
      onCancel={() => {}}
      onSelect={onSelect}
      sessionId="session-fixture"
      t={DEFAULT_THEME}
    />
  )

  const instance = renderSync(element, {
    patchConsole: false,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stderr: stderr as unknown as NodeJS.WriteStream
  })

  try {
    await vi.waitFor(() => expect(stripAnsi(output)).toContain('Välj leverantör'))
    expect(request).toHaveBeenCalledWith('model.options', { session_id: 'session-fixture', include_unconfigured: true })
    expect(stripAnsi(output)).toContain('Fixture Provider')
    input.handler?.('', { return: true })
    instance.rerender(element)
    await vi.waitFor(() => expect(stripAnsi(output)).toContain('Välj modell'))
    expect(stripAnsi(output)).toContain('vendor/model-fixture')
    input.handler?.('', { return: true })
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(
      `vendor/model-fixture --provider provider-fixture ${TUI_SESSION_MODEL_FLAG}`
    )
  } finally {
    instance.unmount()
    instance.cleanup()
    setTuiLanguage('en')
  }
})
