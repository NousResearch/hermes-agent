import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { readClipboardText } from '../lib/clipboard.js'

vi.mock('../lib/clipboard.js', () => ({
  readClipboardText: vi.fn(),
  isUsableClipboardText: (text: unknown) => typeof text === 'string' && text.length > 0
}))

function deferred() {
  let resolve!: (value: any) => void

  const promise = new Promise<any>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function mount(request: any) {
  resetUiState()
  patchUiState({ sid: 'owner', pasteCollapseChars: 5 })
  let composer!: ReturnType<typeof useComposerState>
  const sys = vi.fn()

  function Harness() {
    composer = useComposerState({ gw: { request } as any, submitRef: { current: () => {} }, sys })

    return <Text>{composer.state.input}</Text>
  }

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  return {
    get composer() {
      return composer
    },
    sys,
    close() {
      instance.unmount()
      resetUiState()
    }
  }
}

it('captures hotkey destination before reading the clipboard, while preserving current-owner pastes', async () => {
  vi.stubEnv('SSH_CONNECTION', '')
  vi.stubEnv('SSH_CLIENT', '')
  vi.stubEnv('SSH_TTY', '')
  const request = vi.fn(async () => ({ name: 'shot', path: '/shot.png' }))
  const h = mount(request)

  try {
    const gate = deferred()
    vi.mocked(readClipboardText).mockReturnValueOnce(gate.promise)
    const pending = h.composer.actions.handleTextPaste({ hotkey: true, text: '', value: '', cursor: 0 })
    patchUiState({ sid: 'other' })
    gate.resolve('/shot.png')
    expect(await pending).toBeNull()
    expect(request).not.toHaveBeenCalled()
    expect(h.composer.refs.tokensRef.current).toEqual([])
    vi.mocked(readClipboardText).mockResolvedValueOnce('/shot.png')
    expect(await h.composer.actions.handleTextPaste({ hotkey: true, text: '', value: '', cursor: 0 })).toMatchObject({
      value: '[[ Image 1 ]]'
    })
    expect(request).toHaveBeenCalledWith('image.attach', { path: '/shot.png', session_id: 'other' })
  } finally {
    h.close()
    vi.unstubAllEnvs()
  }
})

it('discards stale image, drop fallback and collapse completions without changing the new composer', async () => {
  for (const mode of ['image', 'clipboard', 'drop', 'collapse']) {
    const gate = deferred()
    const request = vi.fn((method: string) => (method.startsWith('complete.') ? Promise.resolve({}) : gate.promise))
    const h = mount(request)

    try {
      let pending: unknown

      if (mode === 'image') {h.composer.actions.attachImagePath('/shot.png')}
      else if (mode === 'clipboard') {h.composer.actions.attachClipboardImage()}
      else
        {pending = h.composer.actions.handleTextPaste({
          text: mode === 'drop' ? '/file.txt' : 'long paste',
          value: '',
          cursor: 0
        })}

      await Promise.resolve()
      const tokens = h.composer.refs.tokensRef.current.map(t => ({ ...t }))
      patchUiState({ sid: 'other' })
      h.composer.actions.setInput('new draft')
      h.composer.actions.setComposerTokens(tokens)
      gate.resolve(
        mode === 'drop' ? { matched: true, text: 'wrong' } : { name: 'shot', attached: true, path: '/old-path' }
      )
      await pending
      await new Promise<void>(resolve => setImmediate(resolve))
      expect(h.composer.refs.tokensRef.current).toEqual(tokens)
      expect(h.composer.state.input).toBe('new draft')
      expect(request.mock.calls.filter(([method]) => method === 'input.detect_drop')).toHaveLength(0)
      expect(h.sys).not.toHaveBeenCalled()
    } finally {
      h.close()
    }
  }
})
