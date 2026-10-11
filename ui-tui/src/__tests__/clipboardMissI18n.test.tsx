import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { applyLocale, resetLocale } from '../i18n/runtime.js'

afterEach(() => {
  resetLocale()
  resetUiState()
})

it('an explicit /paste with an empty clipboard reports the miss from the active locale pack', async () => {
  resetUiState()
  patchUiState({ sid: 'session' })
  applyLocale('xx', { lang: 'xx', messages: { 'canonical.queue.noClipboardImage': 'XX-no-image' }, surface: 'tui' })
  const sys = vi.fn()
  const request = vi.fn(async () => ({ attached: false }))
  let composer!: ReturnType<typeof useComposerState>

  function Harness() {
    composer = useComposerState({ gw: { request }, submitRef: { current: vi.fn() }, sys } as any)

    return <Text>composer</Text>
  }

  const instance = renderSync(<Harness />, {
    patchConsole: false,
    stderr: new PassThrough() as any,
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 }) as any
  })

  try {
    composer.actions.attachClipboardImage()
    await vi.waitFor(() => expect(sys).toHaveBeenCalledWith('XX-no-image'))
  } finally {
    instance.unmount()
  }
})
