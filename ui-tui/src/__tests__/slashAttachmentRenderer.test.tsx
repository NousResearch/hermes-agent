import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { useSubmission } from '../app/useSubmission.js'

const flush = () => new Promise<void>(resolve => setImmediate(resolve))

function mount(canonical = false, extra: (method: string, params: any) => unknown = () => undefined) {
  const home = mkdtempSync(join(tmpdir(), 'ink-slash-attachment-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'owner' })
  const images = new Map<string, string[]>()
  const pending: Array<{ sid: string; finish: (path: string) => void }> = []

  const request = vi.fn((method: string, params: any) => {
    const handled = extra(method, params)

    if (handled !== undefined) { return Promise.resolve(handled) }

    if (method === 'prompt.submit' && canonical) { return Promise.resolve({ status: 'started' }) }

    if (method === 'image.attach' || method === 'clipboard.paste') {
      return new Promise(resolve => {
        pending.push({ sid: params.session_id, finish(path) {
          images.set(params.session_id, [...(images.get(params.session_id) ?? []), path])
          resolve({ attached: true, name: 'image.png', path })
        } })
      })
    }

    if (method === 'image.detach') {
      images.set(params.session_id, (images.get(params.session_id) ?? []).filter(path => path !== params.path))
    } else if (!method.startsWith('complete.')) {
      throw new Error(`Unexpected RPC: ${method}`)
    }

    return Promise.resolve({})
  })

  const gw = { request, isCanonical: canonical } as any
  const submitRef = { current: (_value: string) => {} }
  const slashRef = { current: (_value: string) => false }
  const slashFlightRef = { current: 0 }
  const sys = vi.fn()
  let composer!: ReturnType<typeof useComposerState>
  let submission!: ReturnType<typeof useSubmission>
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false })
  let output = ''
  stdout.on('data', chunk => { output += chunk.toString() })

  function Harness() {
    composer = useComposerState({ gw, submitRef, sys })
    submission = useSubmission({ gw, submitRef, slashRef, sys, composerActions: composer.actions,
      composerRefs: composer.refs, composerState: composer.state, appendMessage: vi.fn(), setLastUserMsg: vi.fn() })
    slashRef.current = createSlashHandler({ gateway: { gw }, local: {}, transcript: { sys, send: submission.send },
      composer: composer.actions, slashFlightRef } as any)

    return <Text>{composer.state.input}</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any, stdout: stdout as any,
    stderr: new PassThrough() as any, patchConsole: false })

  return {
    home, images, pending, request,
    get composer() { return composer },
    get output() { return output },
    async submit(command: string) {
      composer.actions.setInput(command)
      await flush()
      submission.submit(command)
      await flush()
    },
    close() {
      instance.unmount()
      instance.cleanup()
      resetUiState()
      vi.unstubAllEnvs()
      rmSync(home, { recursive: true, force: true })
    }
  }
}

it('canonical slash images submit staged bytes and deleting tokens removes payload without RPC detach', async () => {
  const h = mount(true)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')
  const path = join(h.home, 'shot.png')
  const bytes = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nL8AAAAASUVORK5CYII=', 'base64')
  writeFileSync(path, bytes)

  try {
    await h.submit(`/image ${path}`)
    await vi.waitFor(() => expect(h.composer.state.input).toContain('[[ Image 1 ]]'))
    const token = h.composer.refs.tokensRef.current[0]!
    expect(token).toMatchObject({ mime: 'image/png' })
    expect(readFileSync(token.path!)).toEqual(bytes)
    await h.submit('caption [[ Image 1 ]]')
    await vi.waitFor(() => expect(h.request.mock.calls.some(([method]) => method === 'prompt.submit')).toBe(true))
    const wire = h.request.mock.calls.find(([method]) => method === 'prompt.submit')![1]
    expect(wire.attachments).toEqual([{ path: token.path, mime: 'image/png' }])
    expect(wire.text).toBe('caption')
    expect(h.request.mock.calls.some(([method]) => method === 'image.attach')).toBe(false)
    await h.submit(`/image ${path}`)
    await vi.waitFor(() => expect(h.composer.refs.tokensRef.current).toHaveLength(1))
    h.composer.actions.syncTokens('')
    expect(h.composer.refs.tokensRef.current).toEqual([])
    expect(h.request.mock.calls.some(([method]) => method === 'image.detach')).toBe(false)
  } finally { h.close() }
})

it('slash attachment submission leaves a visible removable image in the cleared composer', async () => {
  for (const command of ['/image /owned.png', '/paste']) {
    const h = mount()

    try {
      await h.submit(command)
      expect(h.pending).toHaveLength(1)
      expect(h.composer.state.input).toBe('')
      h.pending[0]!.finish('/owned.png')
      await flush()
      expect(h.composer.state.input).toContain('[[ Image 1 ]]')
      expect(h.output).toContain('[[ Image 1 ]]')
      expect(h.composer.refs.tokensRef.current).toEqual([
        expect.objectContaining({ kind: 'image', path: '/owned.png' })
      ])
      expect(h.images.get('owner')).toEqual(['/owned.png'])
      h.composer.actions.syncTokens('')
      h.composer.actions.setInput('')
      await flush()
      expect(h.images.get('owner')).toEqual([])
    } finally { h.close() }
  }
})

it('stale attachment cleanup uses its captured owner and preserves a concurrent valid image', async () => {
  for (const command of ['/image /owned.png', '/paste', 'drop']) {
    for (const navigation of [false, true]) {
      for (const samePath of [false, true]) {
        for (const staleFirst of [false, true]) {
          const h = mount()

          try {
            if (command === 'drop') {
              void h.composer.actions.handleTextPaste({ text: '/owned.png', value: '', cursor: 0 })
            } else {
              await h.submit(command)
            }

            h.composer.actions.setInput('user edit')

            if (navigation) {
              patchUiState({ sid: 'other' })
              h.composer.actions.clearIn()
            }

            await h.submit('/image /valid.png')
            expect(h.pending).toHaveLength(2)
            const validPath = samePath ? '/owned.png' : '/valid.png'
            const first = staleFirst ? 0 : 1
            h.pending[first]!.finish(first === 0 ? '/owned.png' : validPath)
            await flush()
            h.pending[1 - first]!.finish(first === 0 ? validPath : '/owned.png')
            await flush()
            const sid = getUiState().sid!
            expect(h.composer.state.input).toBe('[[ Image 1 ]]')
            expect(h.composer.refs.tokensRef.current).toEqual([
              expect.objectContaining({ kind: 'image', path: validPath })
            ])
            expect(h.images.get(sid)).toContain(validPath)
            const detach = h.request.mock.calls.filter(([method]) => method === 'image.detach')

            if (samePath && !navigation) {
              expect(detach).toEqual([])
            } else {
              expect(detach).toEqual([['image.detach', { session_id: 'owner', path: '/owned.png' }]])
              expect(h.images.get('owner')).not.toContain('/owned.png')
            }
          } finally { h.close() }
        }
      }
    }
  }
})

it('a skill slash command keeps the staged image descriptors through the expanded skill send', async () => {
  const skill = (method: string, params: any) => method === 'slash.exec'
    ? { type: 'skill', name: 'review', message: `SKILL BODY\n\nUser request: ${params.command.replace(/^review\s*/, '')}`, display: `/${params.command}` }
    : undefined
  const h = mount(true, skill)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')
  const path = join(h.home, 'shot.png')
  writeFileSync(path, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nL8AAAAASUVORK5CYII=', 'base64'))

  try {
    await h.submit(`/image ${path}`)
    await vi.waitFor(() => expect(h.composer.state.input).toContain('[[ Image 1 ]]'))
    const token = h.composer.refs.tokensRef.current[0]!
    await h.submit('/review this screenshot [[ Image 1 ]]')
    await vi.waitFor(() => expect(h.request.mock.calls.some(([method]) => method === 'prompt.submit')).toBe(true))
    const wire = h.request.mock.calls.find(([method]) => method === 'prompt.submit')![1]
    expect(wire.attachments).toEqual([{ path: token.path, mime: 'image/png' }])
    expect(wire.text).toContain('SKILL BODY')
    expect(wire.text).not.toContain('[[ Image')
    expect(h.composer.refs.tokensRef.current).toEqual([])
  } finally { h.close() }
})
