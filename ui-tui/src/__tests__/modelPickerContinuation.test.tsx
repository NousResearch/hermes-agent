import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { continuationProviders, ModelPicker } from '../components/modelPicker.js'
import type { GatewayClient } from '../gatewayClient.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const providers = [
  { name: 'Nous free tier', slug: 'nous', free_tier_row: true },
  { name: 'LM Studio', slug: 'lmstudio' },
  { name: 'Local', slug: 'llamacpp', auth_type: 'local' },
  { name: 'Ollama Cloud', slug: 'ollama-cloud', auth_type: 'api_key' },
  { name: 'Anthropic', slug: 'anthropic', auth_type: 'api_key' }
]

describe('ModelPicker continuation variant', () => {
  it('renders all recovery choices even if the provider catalog fails', async () => {
    const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })
    const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
    const stderr = Object.assign(new PassThrough(), { isTTY: false })
    let output = ''
    stdout.on('data', chunk => { output += chunk.toString() })
    const gw = { request: vi.fn(async () => { throw new Error('Catalog unavailable') }) } as unknown as GatewayClient
    const instance = renderSync(
      <ModelPicker
        continuationMessage="Your first 10 tool calls are complete."
        gw={gw} onCancel={vi.fn()} onSelect={vi.fn()} onSignIn={vi.fn()} onSetup={vi.fn()}
        sessionId="focused" t={DEFAULT_THEME}
      />,
      { patchConsole: false, stdout: stdout as unknown as NodeJS.WriteStream, stdin: stdin as unknown as NodeJS.ReadStream, stderr: stderr as unknown as NodeJS.WriteStream }
    )
    try {
      await vi.waitFor(() => expect(stripAnsi(output)).toContain('Catalog unavailable'))
      const frame = stripAnsi(output)
      expect(frame).toContain('Choose how to continue')
      expect(frame).toContain('Sign in or create an account')
      expect(frame).toContain('Use a local model')
      expect(frame).toContain('Other providers')
      expect(frame).toContain('Your conversation is saved.')
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('uses catalog identity to separate local choices from cloud and excludes the exhausted free route', () => {
    expect(continuationProviders(providers, true).map(provider => provider.slug)).toEqual(['lmstudio', 'llamacpp'])
    expect(continuationProviders(providers, false).map(provider => provider.slug)).toEqual(['ollama-cloud', 'anthropic'])
  })
})
