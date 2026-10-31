import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { useComposerState } from '../app/useComposerState.js'

it('a history recall replaces the whole draft, backslash-continued lines included', async () => {
  let composer!: ReturnType<typeof useComposerState>

  function Harness() {
    composer = useComposerState({ gw: { request: vi.fn() } as any, submitRef: { current: () => {} }, sys: vi.fn() })

    return <Text>{composer.state.input}</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  try {
    composer.actions.setInputBuf(['old first line'])
    composer.actions.setInput('old second line')
    await vi.waitFor(() => expect(composer.state.inputBuf).toEqual(['old first line']))

    composer.actions.replaceDraft('recalled prompt')

    await vi.waitFor(() => expect(composer.state.input).toBe('recalled prompt'))
    // Submit joins `[...inputBuf, input]`: a stale buffer line would ride above the recall.
    expect([...composer.state.inputBuf, composer.state.input].join('\n')).toBe('recalled prompt')
  } finally {
    instance.unmount()
  }
})
