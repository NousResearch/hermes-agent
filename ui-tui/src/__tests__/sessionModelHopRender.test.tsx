import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import type { ModelOptionsResult } from '@hermes/shared/gateway-events'
import React from 'react'
import stripAnsi from 'strip-ansi'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SessionModelHop } from '../components/sessionModelHop.js'
import { rememberModelOptions, resetModelOptionsCacheForTests } from '../lib/modelOptionsCache.js'
import { DEFAULT_THEME } from '../theme.js'

afterEach(() => resetModelOptionsCacheForTests())

describe.each([40, 80, 120])('SessionModelHop cached first paint at %i columns', columns => {
  it('reuses a fresh session catalog without another model.options RPC', async () => {
    const result = {
      model: 'nous/hermes-4',
      providers: [
        {
          authenticated: true,
          models: ['hermes-4', 'claude-sonnet-4.6'],
          name: 'Nous Portal',
          slug: 'nous'
        }
      ]
    } as ModelOptionsResult

    rememberModelOptions('sid-1', result)

    const stdout = new PassThrough()
    const stdin = new PassThrough()
    const stderr = new PassThrough()
    const request = vi.fn(() => Promise.resolve(result))

    Object.assign(stdout, { columns, isTTY: false, rows: 30 })
    Object.assign(stdin, { isTTY: false })
    Object.assign(stderr, { isTTY: false })

    const instance = renderSync(
      <SessionModelHop
        gw={{ request } as any}
        maxWidth={columns}
        onCancel={() => {}}
        onOpenProviderPicker={() => {}}
        onSelect={() => {}}
        sessionId="sid-1"
        t={DEFAULT_THEME}
      />,
      {
        patchConsole: false,
        stderr: stderr as NodeJS.WriteStream,
        stdin: stdin as NodeJS.ReadStream,
        stdout: stdout as NodeJS.WriteStream
      }
    )

    for (let i = 0; i < 4; i++) {
      await new Promise(resolve => setTimeout(resolve, 5))
    }

    expect(request).not.toHaveBeenCalled()

    instance.unmount()
    instance.cleanup()
  })
})


it('keeps provider setup reachable when the flat catalog is empty', async () => {
  const result = { model: '', providers: [] } as ModelOptionsResult
  rememberModelOptions('sid-empty', result)

  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 30, isTTY: false })
  const stdin = Object.assign(new PassThrough(), {
    isTTY: true,
    setRawMode: () => {},
    ref: () => {},
    unref: () => {}
  })
  const stderr = new PassThrough()
  const request = vi.fn(() => Promise.resolve(result))
  const onOpenProviderPicker = vi.fn()
  let output = ''

  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const instance = renderSync(
    <SessionModelHop
      gw={{ request } as any}
      maxWidth={80}
      onCancel={() => {}}
      onOpenProviderPicker={onOpenProviderPicker}
      onSelect={() => {}}
      sessionId="sid-empty"
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  try {
    await vi.waitFor(() => expect(output).toContain('no configured models available'))
    stdin.write('\r')
    await vi.waitFor(() => expect(onOpenProviderPicker).toHaveBeenCalledTimes(1))
    expect(request).not.toHaveBeenCalled()
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})

it('preserves q typeahead during a cold model-options load', async () => {
  const result = {
    model: 'nous/hermes-4',
    providers: [
      {
        authenticated: true,
        models: ['hermes-4', 'qwen3-32b'],
        name: 'Nous Portal',
        slug: 'nous'
      }
    ]
  } as ModelOptionsResult

  let resolveRequest: (value: ModelOptionsResult) => void = () => {}
  const request = vi.fn(
    () =>
      new Promise<ModelOptionsResult>(resolve => {
        resolveRequest = resolve
      })
  )
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 30, isTTY: false })
  const stdin = Object.assign(new PassThrough(), {
    isTTY: true,
    setRawMode: () => {},
    ref: () => {},
    unref: () => {}
  })
  const stderr = new PassThrough()
  const onCancel = vi.fn()
  let output = ''

  stdout.on('data', chunk => {
    output += stripAnsi(chunk.toString())
  })

  const instance = renderSync(
    <SessionModelHop
      gw={{ request } as any}
      maxWidth={80}
      onCancel={onCancel}
      onOpenProviderPicker={() => {}}
      onSelect={() => {}}
      sessionId="sid-q"
      t={DEFAULT_THEME}
    />,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  try {
    await vi.waitFor(() => expect(output).toContain('loading session models'))
    stdin.write('q')
    expect(onCancel).not.toHaveBeenCalled()

    resolveRequest(result)

    await vi.waitFor(() => expect(output).toContain('Filter: q'))
    expect(output).toContain('nous/qwen3-32b')
    expect(onCancel).not.toHaveBeenCalled()
    expect(request).toHaveBeenCalledTimes(1)
  } finally {
    instance.unmount()
    instance.cleanup()
  }
})
