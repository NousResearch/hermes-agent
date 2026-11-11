import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'

it('does not restore stale same-session paste results after editing away and back', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-composer-freshness-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'session' })
  let finish!: (value: unknown) => void

  const request = vi.fn((method: string) => method === 'clipboard.paste'
    ? new Promise(resolve => { finish = resolve }) : Promise.resolve({}))

  let composer!: ReturnType<typeof useComposerState>

  function Harness() {
    composer = useComposerState({ gw: { request }, submitRef: { current: vi.fn() }, sys: vi.fn() } as any)

    return <Text>composer</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  try {
    composer.actions.setInput('old')
    const pending = composer.actions.handleTextPaste({ bracketed: true, hotkey: false, text: '', value: 'old', cursor: 3 })
    composer.actions.setInput('new')
    composer.actions.setInput('old')
    finish({ attached: true, name: 'image', path: '/tmp/owned-image.png' })
    expect(await pending).toBeNull()
    expect(composer.refs.tokensRef.current).toEqual([])
    const fresh = composer.actions.handleTextPaste({ bracketed: true, hotkey: false, text: '', value: 'old', cursor: 3 })
    finish({ attached: true, name: 'image', path: '/tmp/owned-image.png' })
    expect(await fresh).toEqual(expect.objectContaining({ value: expect.stringContaining('Image') }))
    expect(composer.refs.tokensRef.current).toHaveLength(1)
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})
