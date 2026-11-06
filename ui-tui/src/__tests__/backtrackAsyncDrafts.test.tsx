import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $pendingRewind } from '../app/backtrackStore.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { takePastes } from '../app/pasteStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { useSubmission } from '../app/useSubmission.js'
import { stageImagePath } from '../lib/imageAttachments.js'

vi.mock('../lib/imageAttachments.js', () => ({ stageClipboardImage: vi.fn(), stageImagePath: vi.fn() }))

const flush = () => new Promise(resolve => setImmediate(resolve))

const settle = async () => {
  for (let i = 0; i < 6; i++) {
    await flush()
  }
}

const block = Array.from({ length: 12 }, (_, i) => `pasted line ${i + 1}`).join('\n')

const rows = [
  { role: 'user', content: 'ALPHA please', text: 'ALPHA please', row_id: 11 },
  { role: 'assistant', content: 'ALPHA', text: 'ALPHA', row_id: 12 },
  { role: 'user', content: 'BETA please', text: 'BETA please', row_id: 13 },
  { role: 'assistant', content: 'BETA', text: 'BETA', row_id: 14 }
]

let home = ''

let unmount = () => {}

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'backtrack-async-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  resetOverlayState()
  $pendingRewind.set(null)
  patchUiState({ busy: false, sid: 'A' })
})

afterEach(() => {
  unmount()
  vi.unstubAllEnvs()
  vi.mocked(stageImagePath).mockReset()
  $pendingRewind.set(null)
  resetUiState()
  resetOverlayState()
  rmSync(home, { force: true, recursive: true })
})

/** The real composer and submit hooks, wired to `/rewind` the way useMainApp wires them. */
function mount(request: (method: string, params: any) => Promise<unknown>) {
  const gw = { isCanonical: true, request: vi.fn(request) } as any
  const sys = vi.fn()
  const send = vi.fn()
  let composer!: ReturnType<typeof useComposerState>
  let submission!: ReturnType<typeof useSubmission>

  function Harness() {
    composer = useComposerState({ gw, submitRef: { current: vi.fn() }, sys } as any)
    submission = useSubmission({
      appendMessage: () => {},
      composerActions: composer.actions,
      composerRefs: composer.refs,
      composerState: composer.state,
      gw,
      setLastUserMsg: () => {},
      slashRef: { current: () => true },
      submitRef: { current: () => {} },
      sys
    } as any)

    return <Text>composer</Text>
  }

  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const instance = renderSync(<Harness />, { patchConsole: false, stdin: new PassThrough() as any, stdout: stdout as any })

  unmount = () => instance.unmount()

  const slash = createSlashHandler({
    slashFlightRef: { current: 0 },
    gateway: { gw, rpc: gw.request },
    transcript: { page: vi.fn(), panel: vi.fn(), send, sys, setHistoryItems: vi.fn(), trimLastExchange: (i: unknown[]) => i },
    local: { getHistoryItems: () => [], getLastUserMsg: () => '', maybeWarn: vi.fn() },
    session: { newSession: vi.fn(), resumeById: vi.fn() },
    composer: {
      appendPaste: (t: string) => composer.actions.appendPaste(t),
      appendText: (t: string) => composer.actions.appendText(t),
      attachImagePath: (p: string) => composer.actions.attachImagePath(p),
      attachImagePaths: (p: string[], o?: any) => composer.actions.attachImagePaths(p, o),
      enqueue: (...args: any[]) => (composer.actions.enqueue as any)(...args),
      restoreDraft: (...args: any[]) => (composer.actions.restoreDraft as any)(...args),
      setInput: (v: any) => composer.actions.setInput(v)
    },
    voice: {}
  } as any)

  return { composer: () => composer, gw, send, slash, submit: () => submission, sys }
}

const resume = { session_id: 'A', revision: 7, execution_generation: 3, messages: rows, info: { model: 'm' } }

describe('a browse paste still resolving belongs to the draft it was typed into', () => {
  const held = () => {
    const releases: Array<() => void> = []

    vi.mocked(stageImagePath).mockImplementation(
      () => new Promise(done => releases.push(() => done({ mime: 'image/png', name: 'shot.png', path: '/cache/shot.png' }) as any))
    )

    return releases
  }

  const rpc = async (method: string) => (method === 'session.resume' ? resume : {})

  it('a session switch while an image path resolves: neither the paste nor the keys behind it land in the new draft', async () => {
    const releases = held()
    const { composer, sys } = mount(rpc)

    composer().actions.setInput('in A')
    const pasted = composer().actions.appendPaste('/shots/shot.png')

    composer().actions.appendText('x')
    await flush()
    patchUiState({ sid: 'B' })
    composer().actions.clearIn()
    composer().actions.setInput('draft in B')
    releases[0]!()
    await pasted
    await settle()

    expect(composer().state.input).toBe('draft in B')
    expect(composer().refs.tokensRef.current).toEqual([])
    expect(sys.mock.calls.map(([text]) => String(text)).join('\n')).toContain('paste dropped')
  })

  it('the draft cleared or submitted in the same session: the paste is dropped, not revived in the next draft', async () => {
    const releases = held()
    const { composer } = mount(rpc)

    composer().actions.setInput('first draft')
    const pasted = composer().actions.appendPaste('/shots/shot.png')

    composer().actions.appendText('y')
    await flush()
    composer().actions.clearIn()
    composer().actions.setInput('second')
    releases[0]!()
    await pasted
    await settle()

    expect(composer().state.input).toBe('second')
  })

  it('backspacing the draft to empty is still the same draft: a resolving paste lands, no notice', async () => {
    const releases = held()
    const { composer, sys } = mount(rpc)

    composer().actions.setInput('ab')
    const pasted = composer().actions.appendPaste('/shots/shot.png')

    composer().actions.setInput('')
    await flush()
    releases[0]!()
    await pasted
    await settle()

    expect(composer().refs.tokensRef.current.map(token => token.kind)).toEqual(['image'])
    expect(sys).not.toHaveBeenCalled()
  })

  it('control: typing on in the same draft keeps the paste and the keys, in order', async () => {
    const releases = held()
    const { composer } = mount(rpc)

    composer().actions.setInput('look')
    const pasted = composer().actions.appendPaste('/shots/shot.png')

    composer().actions.appendText('!')
    await flush()
    releases[0]!()
    await pasted
    await settle()

    expect(composer().state.input).toMatch(/^look.*\[\[ Image 1 \]\].*!$/s)
  })
})

describe('an armed edit through the real submit path', () => {
  const staged = (path: string) => ({ mime: 'image/png', name: 'x.png', path: path.replace('/shots/', '/cache/') })

  const busy = () => new JsonRpcGatewayError('session_busy', { code: 4001, data: { reason: 'session_busy' } })

  async function armWithChips(rewindRefusals: number, fail: () => Promise<never> = async () => Promise.reject(busy())) {
    vi.mocked(stageImagePath).mockImplementation(async path => staged(path) as any)
    let refusals = rewindRefusals

    const harness = mount(async (method: string, params: any) => {
      if (method === 'session.resume') {
        return resume
      }

      if (method === 'session.mutate' && params.operation === 'rewind') {
        if (refusals-- > 0) {
          return fail()
        }

        return { session_id: 'A', operation: 'rewind', revision: 8, execution_generation: 4, rewound_count: 2 }
      }

      return {}
    })

    harness.slash('/rewind')
    await settle()
    const browse = $overlayState.get().rewind!

    browse.onEdit(browse.targets[1]!)
    await settle()
    await harness.composer().actions.appendPaste(block)
    await harness.composer().actions.attachImagePaths(['/shots/a.png', '/shots/b.png'])
    await settle()

    return harness
  }

  it('a refused rewind puts the whole draft back: text with chips, both images and the paste body, still armed', async () => {
    const { composer, send, submit } = await armWithChips(1)
    const display = composer().state.input
    const tokens = [...composer().refs.tokensRef.current]

    expect(tokens.map(token => token.kind)).toEqual(['paste', 'image', 'image'])

    submit().dispatchSubmission(display)
    expect(composer().refs.tokensRef.current).toEqual([]) // cleared before the rewind answers
    await settle()

    expect(send).not.toHaveBeenCalled()
    expect(composer().state.input).toBe(display)
    expect(composer().refs.tokensRef.current).toEqual(tokens)
    expect($pendingRewind.get()?.target.rowId).toBe(13)

    // The retry rewinds and sends both images and the pasted text.
    submit().dispatchSubmission(composer().state.input)
    await settle()

    expect(send).toHaveBeenCalledTimes(1)
    const [text, , shown, , opts] = send.mock.calls[0]!

    expect(text).toContain(block)
    expect(shown).toBe(display)
    expect(opts.attachments.map((a: { path: string }) => a.path)).toEqual(['/cache/a.png', '/cache/b.png'])
  })

  it('a refusal after the user started another draft keeps both, the edit first; a switch away says it was not sent', async () => {
    const { composer, submit, sys } = await armWithChips(2)
    const display = composer().state.input

    submit().dispatchSubmission(display)
    composer().actions.setInput('typed meanwhile')
    await settle()

    expect(composer().state.input).toBe(`${display}\ntyped meanwhile`)
    expect(composer().refs.tokensRef.current.map(token => token.kind)).toEqual(['paste', 'image', 'image'])

    composer().actions.clearIn()
    composer().actions.setInput(display)
    await flush()
    submit().dispatchSubmission(display)
    patchUiState({ sid: 'B' })
    await settle()

    expect(composer().state.input).toBe('')
    expect(sys.mock.calls.map(([text]) => String(text)).join('\n')).toContain('edit not sent')
  })

  it('the sent paste chip keeps its body, so the transcript can open it (Ctrl+P / click)', async () => {
    const { composer, send, submit } = await armWithChips(0)

    submit().dispatchSubmission(composer().state.input)
    await settle()

    const [text, , shown] = send.mock.calls[0]!
    const pastes = takePastes(text)

    expect(pastes?.display).toBe(shown)
    expect(pastes?.pastes.map(paste => paste.text)).toEqual([block])
  })

  it('Esc Esc while the edit is still sending: its refusal brings the draft back but never re-arms it', async () => {
    let refuse!: () => void
    const { composer, submit } = await armWithChips(1, () => new Promise((_, reject) => (refuse = () => reject(busy()))))
    const display = composer().state.input

    submit().dispatchSubmission(display)
    await flush()
    composer().actions.setInput(' ')
    composer().actions.setInput('')
    $overlayState.set({ ...$overlayState.get(), rewind: null })
    const { cancelPendingRewind } = await import('../app/backtrackStore.js')

    cancelPendingRewind()
    refuse()
    await settle()

    expect(composer().state.input).toBe(display)
    expect($pendingRewind.get()).toBeNull()
  })

  it('a transport error after a switch away never claims nothing was removed', async () => {
    let fail!: () => void
    const { submit, sys } = await armWithChips(1, () => new Promise((_, reject) => (fail = () => reject(new Error('socket closed')))))

    submit().dispatchSubmission('edited')
    await flush()
    patchUiState({ sid: 'B' })
    fail()
    await settle()

    const said = sys.mock.calls.map(([text]) => String(text)).join('\n')

    expect(said).toContain('may have been applied')
    expect(said).not.toContain('nothing was removed')
  })
})
