import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { ModelPicker } from '../components/modelPicker.js'
import { DEFAULT_THEME } from '../theme.js'

const flush = async () => {
  await new Promise(resolve => setImmediate(resolve))
  await new Promise(resolve => setImmediate(resolve))
}

// A lone ESC byte is held ~50ms by the input parser (incomplete-sequence
// timeout) before it dispatches as key.escape — outlast that window.
const flushEsc = async () => {
  await new Promise(resolve => setTimeout(resolve, 80))
  await flush()
}

// Behaviour contract for #157542: wherever the picker captures text (list
// filters, API key entry) 'q' is text, never a quit shortcut.
function harness(requestImpl?: (method: string) => Promise<unknown>) {
  const request = vi.fn(
    requestImpl ??
      (async (method: string) => {
    if (method !== 'model.options') {
      throw new Error('invalid_params')
    }

    return {
      model: 'qwen/qwen3-max',
      providers: [
        {
          authenticated: true,
          is_current: true,
          models: ['qwen/qwen3-max', 'qwen/qwen3-coder', 'claude-sonnet-4'],
          name: 'Nous Portal',
          slug: 'nous',
          total_models: 3
        },
        {
          authenticated: true,
          models: ['gpt-5.6'],
          name: 'OpenAI',
          slug: 'openai',
          total_models: 1
        }
      ]
    }
      })
  )

  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
  let output = ''
  const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })
  const onCancel = vi.fn()
  const onSelect = vi.fn()

  // Frames accumulate on the stream; take() returns only what was painted
  // since the last take, so "no longer shows X" assertions see the live frame.
  const take = () => {
    const frame = output
    output = ''

    return frame
  }

  const element = (
    <ModelPicker
      gw={{ request } as any}
      onCancel={onCancel}
      onSelect={onSelect}
      sessionId={null}
      t={DEFAULT_THEME}
    />
  )

  const instance = renderSync(element, {
    stdin: stdin as any,
    stdout: stdout as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  return { instance, stdin, onCancel, onSelect, take, element }
}

describe('ModelPicker q is filter text on list stages (#157542)', () => {
  it('typing q as the first filter character on the model stage filters instead of closing', async () => {
    const h = harness()

    try {
      await flush()
      h.instance.rerender(h.element)
      await flush()
      // Enter on the current provider (Nous Portal) opens the model stage.
      h.stdin.write('\r')
      await flush()
      expect(h.take()).toContain('qwen/qwen3-coder')

      h.stdin.write('q')
      await flush()
      expect(h.onCancel).not.toHaveBeenCalled()
      expect(h.take()).toContain('filter: q')

      // A second q extends the query instead of quitting mid-search.
      h.stdin.write('w')
      await flush()
      expect(h.onCancel).not.toHaveBeenCalled()
      expect(h.take()).toContain('filter: qw')
      expect(h.onSelect).not.toHaveBeenCalled()
    } finally {
      h.instance.unmount()
    }
  })

  it('typing q while entering an API key does not close the picker', async () => {
    // Sibling path: the key stage is not a list stage, so the shared overlay
    // q-handler used to stay active and quit mid-key-entry (#157542).
    const request = vi.fn(async (method: string) => {
      if (method !== 'model.options') {
        throw new Error('invalid_params')
      }

      return {
        model: '',
        providers: [
          {
            auth_type: 'api_key',
            authenticated: false,
            key_env: 'NOUS_API_KEY',
            models: [],
            name: 'Nous Portal',
            slug: 'nous',
            total_models: 0
          }
        ]
      }
    })

    const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
    let output = ''
    const stdout = Object.assign(new PassThrough(), { columns: 100, rows: 30, isTTY: false })
    stdout.on('data', chunk => {
      output += chunk.toString()
    })

    const take = () => {
      const frame = output
      output = ''

      return frame
    }

    const onCancel = vi.fn()

    const element = (
      <ModelPicker
        gw={{ request } as any}
        onCancel={onCancel}
        onSelect={vi.fn()}
        sessionId={null}
        t={DEFAULT_THEME}
      />
    )

    const instance = renderSync(element, {
      stdin: stdin as any,
      stdout: stdout as any,
      stderr: new PassThrough() as any,
      patchConsole: false
    })

    try {
      await flush()
      instance.rerender(element)
      await flush()
      // Enter on the unconfigured api_key provider opens the key-entry stage.
      stdin.write('\r')
      await flush()
      expect(take()).toContain('NOUS_API_KEY')

      stdin.write('q')
      await flush()
      expect(onCancel).not.toHaveBeenCalled()
      // The 'q' landed in the key input (one masked bullet), not a quit.
      expect(take()).toContain('•')

      // Esc still backs out of key entry to the provider list.
      stdin.write('\x1b')
      await flushEsc()
      expect(onCancel).not.toHaveBeenCalled()
      expect(take()).toContain('Select provider')
    } finally {
      instance.unmount()
    }
  })
})
