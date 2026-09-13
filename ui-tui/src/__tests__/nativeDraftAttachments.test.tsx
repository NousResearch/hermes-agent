import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React, { useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerInput } from '../app/useComposerInput.js'
import { useComposerState } from '../app/useComposerState.js'
import { useInputHandlers } from '../app/useInputHandlers.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { useSubmission } from '../app/useSubmission.js'
import { TextInput } from '../components/textInput.js'
import { DEFAULT_VOICE_RECORD_KEY } from '../lib/platform.js'

class Input extends EventEmitter {
  chunks: string[] = []
  isTTY = true
  isRaw = false
  readableLength = 0
  ref = vi.fn()
  unref = vi.fn()
  setEncoding = vi.fn()
  setRawMode = vi.fn((enabled: boolean) => { this.isRaw = enabled })
  read() { const next = this.chunks.shift() ?? null; this.readableLength = this.chunks.length;

 return next }
  send(...chunks: string[]) { this.chunks.push(...chunks); this.readableLength = this.chunks.length; this.emit('readable') }
}

class Gateway extends EventEmitter {
  states: any[] = []
  results: any[] = []
  stage = Promise.resolve<any>(null)
  request = vi.fn((method: string) => method === 'file.attach' ? this.stage : Promise.resolve({ items: [] }))
  publishDraftState = (state: any) => { this.states.push(state) }
  publishDraftResult = (result: any) => { this.results.push(result) }
}
const settle = () => new Promise(resolve => setTimeout(resolve, 30))

const deferred = () => { let resolve!: (value: any) => void; let reject!: (error: Error) => void; const promise = new Promise<any>((a, b) => { resolve = a; reject = b });

 return { promise, resolve, reject } }

const cleanups: (() => void)[] = []
afterEach(() => { cleanups.splice(0).forEach(fn => fn()); resetOverlayState(); resetUiState() })

async function mount(realSubmit = false, tty = false) {
  patchUiState({ sid: 'runtime-a' })
  const gw = new Gateway()
  const stdin = new Input()
  const stdout = new PassThrough()
  const stderr = new PassThrough()
  Object.assign(stdout, { columns: 100, rows: 24, isTTY: tty })
  let composer: ReturnType<typeof useComposerState>
  const submits: string[] = []
  const sys = vi.fn()
  const appendMessage = vi.fn()
  const setLastUserMsg = vi.fn()
  let submission: ReturnType<typeof useSubmission>
  let session: ReturnType<typeof useSessionLifecycle>

  function Harness() {
    const submitRef = useRef(() => {})
    const slashRef = useRef<any>(() => false)
    const slashFlightRef = useRef(0)
    composer = useComposerState({ gw: gw as any, submitRef, sys })
    submission = useSubmission({ composerActions: composer.actions, composerRefs: composer.refs, composerState: composer.state,
      gw: gw as any, submitRef, slashRef, sys, appendMessage, setLastUserMsg })
    session = useSessionLifecycle({ composerActions: composer.actions, gw, rpc: gw.request,
      colsRef: useRef(100), scrollRef: useRef(null), setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(),
      setStickyPrompt: vi.fn(), setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn(), sys: vi.fn(), panel: vi.fn() } as any)
    slashRef.current = createSlashHandler({ gateway: { gw }, local: { catalog: null }, slashFlightRef,
      composer: composer.actions, transcript: { send: submission.send, sys: vi.fn(), page: vi.fn() } } as any)
    useInputHandlers({
      composer, gateway: { gw, rpc: gw.request },
      actions: { dispatchSubmission: submission.dispatchSubmission, sys: vi.fn(), die: vi.fn(), guardBusySessionSwitch: () => false, newSession: vi.fn() },
      terminal: { stdout, scrollRef: { current: null }, hasSelection: false, selection: { clearSelection: vi.fn(), copySelection: vi.fn() }, scrollWithSelection: vi.fn() },
      voice: { recordKey: DEFAULT_VOICE_RECORD_KEY }, wheelStep: 3
    } as any)
    const updateInput = useComposerInput(composer.actions, composer.refs, composer.state.historyIdx)

    return <TextInput columns={100} onChange={updateInput}
      onHandle={composer.actions.setNativeInput}
      onPaste={composer.actions.handleTextPaste}
      onSubmit={value => { submits.push(value);

 if (realSubmit) {submission.submit(value);} else {composer.actions.clearIn()} }}
      value={composer.state.input} />
  }

  const instance = renderSync(<Harness />, { patchConsole: false, stdin: stdin as any, stdout: stdout as any, stderr: stderr as any })
  cleanups.push(() => { instance.unmount(); instance.cleanup() })
  await settle()

  const request = (id = 'request-a') => {
    const state = gw.states.at(-1)
    expect(state?.available).toBe(true)

    return { type: 'draft.attach', request_id: id, expected: { session_id: state.session_id, draft_id: state.draft_id, pty_instance: 'pty-a', connection_generation: 1 }, path: '/upload/notes.pdf' }
  }

  return { gw, stdin, submits, sys, appendMessage, setLastUserMsg, composer: () => composer!, submission: () => submission!, session: () => session!, request }
}

describe('automatic native draft attachments', () => {
  it.each(['path', 'clipboard', 'drop', 'empty paste'])(
    'rejects an excess native image via %s without evicting admitted metadata', async mode => {
      const h = await mount()
      let image = 0
      h.gw.request.mockImplementation((method: string) => {
        if (method === 'image.attach' || method === 'clipboard.paste') {
          const path = `/native/${++image}.png`

          return Promise.resolve({ attached: true, name: `${image}.png`, path })
        }

        if (method === 'input.detect_drop') {
          return Promise.resolve({ matched: true, is_image: true, path: '/native/excess.png', text: '[User attached image: excess.png]' })
        }

        return Promise.resolve({ items: [] })
      })

      for (let i = 0; i < 32; i++) {
        h.composer().actions.attachImagePath(`/native/${i + 1}.png`)
        await settle()
      }

      h.stdin.send(' leftRIGHT', '\x1b[D', '\x1b[D', '\x1b[D', '\x1b[D', '\x1b[D')
      await settle()
      const admitted = [...h.composer().refs.tokensRef.current]
      const draft = h.composer().state.input
      expect(admitted).toHaveLength(32)

      if (mode === 'path') { h.composer().actions.attachImagePath('/native/excess.png') }
      else if (mode === 'clipboard') { h.composer().actions.attachClipboardImage() }
      else { h.stdin.send(`\x1b[200~${mode === 'drop' ? '/native/excess.png' : ''}\x1b[201~`) }

      await settle()
      expect(h.composer().refs.tokensRef.current).toEqual(admitted)
      expect(h.composer().state.input).toBe(draft)
      expect(h.sys).toHaveBeenCalledWith(expect.stringMatching(/32.*image|image.*32/i))
      expect(h.gw.request.mock.calls.filter(([method]) => method === 'image.detach')).toEqual([])
      expect(image).toBe(32)

      if (mode === 'drop') {
        expect(h.gw.request.mock.calls.filter(([method]) => method === 'input.detect_drop')).toEqual([
          ['input.detect_drop', { session_id: 'runtime-a', text: '/native/excess.png', attach_image: false }]
        ])
      }

      h.stdin.send('!')
      await settle()
      expect(h.composer().state.input).toBe(`${draft.slice(0, -5)}!RIGHT`)

      if (mode === 'drop') {
        const generic = deferred()
        h.gw.request.mockImplementation((method: string) => method === 'input.detect_drop' ? generic.promise : Promise.resolve({}))
        h.stdin.send('\x1b[200~/notes.txt\x1b[201~')
        await settle()
        generic.resolve({ matched: true, is_image: false, text: '[User attached file: /notes.txt]' })
        await settle()
        expect(h.composer().state.input).toContain('[User attached file: /notes.txt]')
        expect(h.composer().refs.tokensRef.current).toEqual(admitted)
      } else {
        h.stdin.send('\x01', '\x0b')
        await settle()
        h.composer().actions.attachClipboardImage()
        await settle()
        expect(h.composer().refs.tokensRef.current.map(token => token.path)).toEqual(['/native/33.png'])
      }
    }
  )

  it.each(['complete', 'fail'])('reserves native slots through racing %s completions without losing live input', async outcome => {
    const h = await mount()
    let image = 0
    h.gw.request.mockImplementation((method: string) => {
      if (method === 'image.attach') {return Promise.resolve({ name: 'seed.png', path: `/native/${++image}.png` })}

      return Promise.resolve({ items: [] })
    })

    for (let i = 0; i < 30; i++) {
      h.composer().actions.attachImagePath(`/native/${i}.png`)
      await settle()
    }

    const first = deferred()
    const second = deferred()
    h.gw.request.mockImplementation((method: string, params?: any) => {
      if (method === 'image.attach') {
        return params.path === '/first.png' ? first.promise : outcome === 'fail' ? Promise.reject(new Error('try detector')) : second.promise
      }

      if (method === 'input.detect_drop') {return second.promise}

      if (method === 'clipboard.paste') {return Promise.resolve({ attached: true, path: '/retry.png' })}

      return Promise.resolve({ items: [] })
    })
    h.composer().actions.attachImagePath('/first.png')
    h.stdin.send('\x1b[200~/second.png\x1b[201~')
    await settle()
    h.composer().actions.attachClipboardImage()
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'clipboard.paste')).toEqual([])
    h.stdin.send(' caption')
    await settle()
    second.resolve({ name: 'second.png', path: '/second.png', matched: true, is_image: true, remainder: 'drop-caption' })
    await settle()

    if (outcome === 'complete') {first.resolve({ name: 'first.png', path: '/first.png' })}
    else {first.reject(new Error('image unavailable'))}

    await settle()

    if (outcome === 'fail') {
      h.composer().actions.attachClipboardImage()
      await settle()
    }

    const tokens = h.composer().refs.tokensRef.current
    expect(tokens).toHaveLength(32)
    expect(tokens.slice(-2).map(token => token.path)).toEqual(['/second.png', outcome === 'complete' ? '/first.png' : '/retry.png'])
    expect(h.composer().state.input).toContain('caption')
    expect(h.composer().state.input).toContain('drop-caption')

    for (const token of tokens) {expect(h.composer().state.input).toContain(token.label)}
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'image.detach')).toEqual([])
  })

  it('does not trim admitted images when adding paste/legacy tokens and can retry after freeing capacity', async () => {
    const h = await mount(true)
    const paths = Array.from({ length: 32 }, (_, i) => `/staged/image-${i}.png`)

    for (const [i, path] of paths.entries()) {
      h.gw.stage = Promise.resolve({ path, ref_text: `@file:${path}`, image: { name: `${i}.png`, mime_type: 'image/png' } })
      h.gw.emit('draft.request', h.request(`image-${i}`))
    }

    await settle()
    patchUiState({ pasteCollapseLines: 2, pasteCollapseChars: 0 })
    h.stdin.send('\x1b[200~first\nsecond\x1b[201~')
    await settle()
    expect(h.composer().refs.tokensRef.current.filter(token => token.kind === 'image').map(token => token.path)).toEqual(paths)
    expect(h.composer().refs.tokensRef.current.at(-1)).toMatchObject({ kind: 'paste', text: 'first\nsecond' })
    h.gw.request.mockImplementation((method: string) => {
      if (method === 'file.attach') {return h.gw.stage}

      if (method === 'image.attach') {return Promise.resolve({ name: 'legacy.png', path: '/legacy.png' })}

      return Promise.resolve({ items: [] })
    })
    h.composer().actions.attachImagePath('/legacy.png')
    await settle()
    expect(h.composer().refs.tokensRef.current.filter(token => token.kind === 'image').map(token => token.path)).toEqual([...paths, '/legacy.png'])
    h.gw.stage = Promise.resolve({ path: '/staged/retry.png', ref_text: '@file:/staged/retry.png', image: { name: 'retry.png', mime_type: 'image/png' } })
    const req = h.request('retry')
    h.gw.emit('draft.request', req)
    await settle()
    expect(h.gw.results.at(-1).status).toBe('failed')
    h.stdin.send('\x01', '\x0b')
    await settle()
    expect(h.composer().refs.tokensRef.current).toEqual([])
    h.gw.emit('draft.request', req)
    await settle()
    expect(h.gw.results.at(-1)).toMatchObject({ status: 'attached', path: '/staged/retry.png' })
    h.stdin.send('\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([
      ['prompt.submit', { session_id: 'runtime-a', text: '', draft_image_paths: ['/staged/retry.png'] }]
    ])
  })

  it('rejects an excess image before insertion/ACK and submits all 32 admitted images', async () => {
    const h = await mount(true)
    const paths = Array.from({ length: 32 }, (_, i) => `/staged/image-${i}.png`)

    for (const [i, path] of [...paths, '/staged/excess.png'].entries()) {
      h.gw.stage = Promise.resolve({ path, ref_text: `@file:${path}`, image: { name: `${i}.png`, mime_type: 'image/png' } })
      h.gw.emit('draft.request', h.request(`image-${i}`))
    }

    await settle()
    expect(h.gw.results.slice(0, 32).map(result => [result.status, result.path])).toEqual(paths.map(path => ['attached', path]))
    expect(h.gw.results.at(-1)).toMatchObject({ status: 'failed', request_id: 'image-32', error: expect.stringMatching(/32/) })
    expect(h.composer().state.input).not.toContain('[[ Image 33 ]]')
    expect(h.composer().refs.tokensRef.current.map(token => token.path)).toEqual(paths)
    h.stdin.send(' caption', '\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([
      ['prompt.submit', { session_id: 'runtime-a', text: 'caption', draft_image_paths: paths }]
    ])
    expect(h.gw.request.mock.calls.filter(([method]) => method.startsWith('image.'))).toEqual([])
  })

  it.each([
    ['direct', 'session'], ['queued', 'session'], ['direct', 'profile'], ['queued', 'profile']
  ])('binds %s interpolation before awaiting shell output across a %s switch', async (mode, scope) => {
    const h = await mount(true)
    h.gw.stage = Promise.resolve({ path: '/staged/a.png', ref_text: '@file:/staged/a.png', image: { name: 'a.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()
    const shell = deferred()
    const drop = deferred()
    const submit = deferred()
    h.gw.request.mockImplementation((method: string) => {
      if (method === 'shell.exec') {return shell.promise}

      if (method === 'input.detect_drop') {return drop.promise}

      if (method === 'prompt.submit') {return submit.promise}

      if (method === 'file.attach') {return h.gw.stage}

      return Promise.resolve({ items: [] })
    })

    if (mode === 'queued') {patchUiState({ busy: true, busyInputMode: 'queue' })}
    h.submission().dispatchSubmission('{!slow-command} [[ Image 1 ]]')

    if (mode === 'queued') {
      const item = h.composer().actions.dequeue()!
      expect(item.draftImages?.map(image => image.path)).toEqual(['/staged/a.png'])
      patchUiState({ busy: false })
      h.submission().sendQueued(item)
    }

    expect(h.gw.request.mock.calls.filter(([method]) => method === 'shell.exec')).toHaveLength(1)
    patchUiState({ busy: true, status: 'B working', ...(scope === 'session'
      ? { sid: 'runtime-b' }
      : { info: { model: 'test', profile_name: 'other', tools: {}, skills: {} } }) })
    h.gw.stage = Promise.resolve({ path: '/staged/b.png', ref_text: '@file:/staged/b.png', image: { name: 'b.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request('b'))
    await settle()
    h.stdin.send(' B draft')
    await settle()
    const bState = getUiState()
    const bInput = h.composer().state.input
    const bTokens = h.composer().refs.tokensRef.current
    shell.resolve({ stdout: 'resolved A', stderr: '', code: 0 })
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'input.detect_drop')).toEqual([
      ['input.detect_drop', { session_id: 'runtime-a', text: 'resolved A' }]
    ])
    expect(getUiState()).toEqual(bState)
    drop.resolve({ matched: false })
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([
      ['prompt.submit', { session_id: 'runtime-a', text: 'resolved A', draft_image_paths: ['/staged/a.png'] }]
    ])

    if (mode === 'direct') {submit.reject(new Error('session busy'))}
    else {submit.resolve({ voice_stopped: true })}

    await settle()
    expect(getUiState()).toEqual(bState)
    expect(h.composer().state.input).toBe(bInput)
    expect(h.composer().refs.tokensRef.current).toEqual(bTokens)
    expect(h.composer().refs.queueRef.current).toEqual([])
    expect(h.appendMessage).not.toHaveBeenCalled()
    expect(h.setLastUserMsg).not.toHaveBeenCalled()
    expect(h.sys).not.toHaveBeenCalled()
  })

  it('keeps history recall separate from edits and resets undo/redo across programmatic replacement', async () => {
    const h = await mount(false, true)
    h.composer().refs.historyRef.current.splice(0, Infinity, 'older', 'newer')
    h.gw.request.mockImplementation((method: string) => method === 'image.attach'
      ? Promise.resolve({ name: 'path.png', path: '/legacy/path.png' }) : Promise.resolve({ items: [] }))
    h.composer().actions.attachImagePath('/legacy/path.png')
    await settle()
    h.stdin.send(' caption')
    await settle()
    h.stdin.send('\x1b[A')
    await settle()
    expect(h.composer().state.input).toBe('newer')
    expect(h.composer().state.historyIdx).toBe(1)
    h.stdin.send('\x1b[A')
    await settle()
    expect(h.composer().state.input).toBe('older')
    expect(h.composer().state.historyIdx).toBe(0)
    h.stdin.send('\x1a', '\x19')
    await settle()
    expect(h.composer().state.input).toBe('older')
    h.stdin.send('\x1b[B')
    await settle()
    h.stdin.send('\x1b[B')
    await settle()
    expect(h.composer().state.input).toBe('[[ Image 1 ]] caption')
    expect(h.composer().state.historyIdx).toBeNull()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'image.detach')).toEqual([])
    h.stdin.send('\x1b[A')
    await settle()
    h.stdin.send(' edited')
    await settle()
    expect(h.composer().state.historyIdx).toBeNull()
    expect(h.composer().refs.historyDraftRef.current).toBe('newer edited')
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'image.detach')).toEqual([
      ['image.detach', { path: '/legacy/path.png', session_id: 'runtime-a' }]
    ])
    // A not-yet-flushed fast echo may not overwrite a replacement later.
    h.stdin.send('x')
    h.composer().actions.setInput('replacement')
    h.stdin.send('!')
    await settle()
    expect(h.composer().state.input).toBe('replacement!')
    h.stdin.send('\x1a')
    await settle()
    expect(h.composer().state.input).toBe('replacement')
    h.stdin.send('\x1a')
    await settle()
    expect(h.composer().state.input).toBe('replacement')
  })

  it('preserves legacy path/clipboard images until submit and makes the cleared input an undo boundary', async () => {
    const h = await mount(true, true)
    h.gw.request.mockImplementation((method: string) => {
      if (method === 'file.attach') {return h.gw.stage}

      if (method === 'image.attach') {return Promise.resolve({ name: 'path.png', path: '/legacy/path.png' })}

      if (method === 'clipboard.paste') {return Promise.resolve({ attached: true, name: 'clip.png', path: '/legacy/clip.png' })}

      return Promise.resolve({ items: [] })
    })
    h.composer().actions.attachImagePath('/legacy/path.png')
    await settle()
    h.composer().actions.attachClipboardImage()
    await settle()
    h.gw.stage = Promise.resolve({ path: '/staged/browser.png', ref_text: '@file:/staged/browser.png', image: { name: 'browser.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()
    expect(h.gw.results.at(-1).status).toBe('attached')
    expect(h.composer().refs.tokensRef.current.map(token => token.path)).toEqual([
      '/legacy/path.png', '/legacy/clip.png', '/staged/browser.png'
    ])
    h.stdin.send(' caption', '\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => ['image.attach', 'clipboard.paste', 'image.detach', 'prompt.submit'].includes(method))).toEqual([
      ['image.attach', { path: '/legacy/path.png', session_id: 'runtime-a' }],
      ['clipboard.paste', { session_id: 'runtime-a' }],
      ['prompt.submit', { session_id: 'runtime-a', text: 'caption', draft_image_paths: ['/staged/browser.png'] }]
    ])
    expect(h.composer().state.input).toBe('')
    h.stdin.send('\x1a')
    await settle()
    expect(h.composer().state.input).toBe('')
    patchUiState({ busy: false })
    h.stdin.send('next', '\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit').at(-1)).toEqual([
      'prompt.submit', { session_id: 'runtime-a', text: 'next' }
    ])
  })

  it.each(['', 'inspect this'])('submits a completed file reference with one Enter after typing %j', async text => {
    const h = await mount(true, true)
    const ref = '@file:/staged/notes.pdf'
    h.gw.stage = Promise.resolve({ path: '/staged/notes.pdf', ref_text: ref })
    h.gw.request.mockImplementation((method: string) => {
      if (method === 'file.attach') {return h.gw.stage}

      // Path completion rewrites an absolute reference relative to the workspace.
      return Promise.resolve({ items: method === 'complete.path'
        ? [{ text: '@file:../staged/notes.pdf', display: 'notes.pdf', meta: 'file' }] : [] })
    })
    const req = h.request()
    h.gw.emit('draft.request', req)
    await settle()
    expect(h.gw.results).toEqual([expect.objectContaining({
      status: 'attached', request_id: req.request_id, identity: req.expected, label: ref
    })])
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([])
    // Let the real useCompletion debounce run before pressing Enter.
    await new Promise(resolve => setTimeout(resolve, 100))

    if (text) {
      h.stdin.send(text)
      await settle()
      expect(h.composer().state.input).toBe(`${ref} ${text}`)
    }

    h.stdin.send('\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([
      ['prompt.submit', { session_id: 'runtime-a', text: `${ref} ${text}`.trimEnd() }]
    ])
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'complete.path')).toEqual([])
  })

  it('uses fast-echo input refs before the parent React value has caught up', async () => {
    const h = await mount(false, true)
    const stage = deferred(); h.gw.stage = stage.promise
    h.gw.emit('draft.request', h.request())
    h.stdin.send('unflushed')
    expect(h.composer().state.input).toBe('')
    stage.resolve({ path: '/staged/a', ref_text: '@file:/staged/a' })
    await settle()
    expect(h.composer().state.input).toBe('unflushed @file:/staged/a ')
    expect(h.gw.results.at(-1).status).toBe('attached')
  })

  it('allocates image labels in the shared native namespace, including existing unbound text tokens', async () => {
    const h = await mount()
    h.composer().actions.setInput('[[ Image 1 ]] [[ Image 2 ]]')
    h.composer().actions.setComposerTokens([{ index: 1, kind: 'image', label: '[[ Image 1 ]]', path: '/legacy/image.png' }])
    h.gw.stage = Promise.resolve({ path: '/staged/three.png', ref_text: '@file:/staged/three.png', image: { name: 'three.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()
    expect(h.gw.results.at(-1).label).toBe('[[ Image 3 ]]')
    expect(h.composer().refs.tokensRef.current.map(token => token.label)).toEqual(['[[ Image 1 ]]', '[[ Image 3 ]]'])
  })

  it.each(['newSession', 'activateLiveSession', 'resumeById'] as const)('invalidates before %s awaits the backend', async action => {
    const h = await mount()
    const pending = deferred(); h.gw.stage = pending.promise
    const req = h.request()
    h.gw.emit('draft.request', req)
    h.gw.request.mockImplementation((method: string) => method === 'setup.status' ? Promise.resolve({ provider_configured: false }) : Promise.resolve(null))
    void h.session()[action]('replacement')
    expect(h.gw.states.at(-1).draft_id).not.toBe(req.expected.draft_id)
    pending.resolve({ path: '/staged/stale', ref_text: '@file:/staged/stale' })
    await settle()
    expect(h.gw.results.at(-1).status).toBe('stale')
    expect(h.composer().state.input).toBe('')
  })

  it.each(['session', 'profile'])('does not carry admitted image metadata across a %s replacement', async destination => {
    const h = await mount(true)
    h.gw.stage = Promise.resolve({ path: '/staged/one.png', ref_text: '@file:/staged/one.png', image: { name: 'one.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()
    const old = h.request('old')

    if (destination === 'session') {patchUiState({ sid: 'runtime-b' })}
    else {patchUiState({ info: { model: 'test', profile_name: 'other', tools: {}, skills: {} } })}

    expect(h.gw.states.at(-1).draft_id).not.toBe(old.expected.draft_id)
    expect(h.composer().refs.tokensRef.current).toEqual([])
  })

  it('carries admitted images through an async skill dispatch without reading the next composer', async () => {
    const h = await mount(true)
    h.gw.stage = Promise.resolve({ path: '/staged/one.png', ref_text: '@file:/staged/one.png', image: { name: 'one.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()
    const skill = deferred()
    h.gw.request.mockImplementation((method: string) => method === 'slash.exec' ? skill.promise : Promise.resolve({}))
    h.submission().dispatchSubmission('/attachment-test-skill caption [[ Image 1 ]]')
    h.stdin.send('next draft')
    skill.resolve({ type: 'skill', name: 'attachment-test-skill', message: 'Inspect carefully: caption [[ Image 1 ]]', display: '/attachment-test-skill caption [[ Image 1 ]]' })
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit').at(-1)).toEqual([
      'prompt.submit', { session_id: 'runtime-a', text: 'Inspect carefully: caption', draft_image_paths: ['/staged/one.png'] }
    ])
    expect(h.composer().state.input).toBe('next draft')
  })

  it('queue replacement retires old staging, restores its own images, and submits only surviving/new tokens', async () => {
    const h = await mount(true)
    h.gw.stage = Promise.resolve({ path: '/staged/one.png', ref_text: '@file:/staged/one.png', image: { name: 'one.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()
    patchUiState({ busy: true, busyInputMode: 'queue' })
    h.submission().dispatchSubmission('caption [[ Image 1 ]]')
    await settle()
    const pending = deferred(); h.gw.stage = pending.promise
    const old = h.request('old')
    h.gw.emit('draft.request', old)
    h.stdin.send('\x1b[A')
    await settle()
    expect(h.composer().state.input).toBe('caption [[ Image 1 ]]')
    expect(h.composer().refs.tokensRef.current).toEqual([expect.objectContaining({ source: 'draft', path: '/staged/one.png' })])
    pending.resolve({ path: '/staged/stale', ref_text: '@file:/staged/stale' })
    await settle()
    expect(h.gw.results.at(-1).status).toBe('stale')
    h.gw.stage = Promise.resolve({ path: '/staged/two.png', ref_text: '@file:/staged/two.png', image: { name: 'two.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request('new'))
    await settle()
    h.composer().actions.setInput('edited [[ Image 2 ]]')
    await settle()
    patchUiState({ busy: false })
    h.stdin.send('\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit').at(-1)).toEqual([
      'prompt.submit', { session_id: 'runtime-a', text: 'edited', draft_image_paths: ['/staged/two.png'] }
    ])
    expect(h.composer().refs.queueRef.current).toEqual([])
  })

  it.each(['queue', 'steer', 'interrupt', 'slash', 'retry'])('keeps image metadata with the exact %s submission, including queue drain', async mode => {
    const h = await mount(true)
    h.gw.stage = Promise.resolve({ path: '/staged/one.png', ref_text: '@file:/staged/one.png', image: { name: 'one.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()

    if (mode === 'retry') {
      let rejected = false
      h.gw.request.mockImplementation((method: string) => {
        if (method === 'prompt.submit' && !rejected) { rejected = true;

 return Promise.reject(new Error('session busy')) }

        return Promise.resolve({})
      })
    }

    patchUiState({ busy: mode !== 'slash' && mode !== 'retry', busyInputMode: mode === 'steer' ? 'steer' : mode === 'interrupt' ? 'interrupt' : 'queue' })
    h.submission().dispatchSubmission(`${mode === 'slash' ? '/queue ' : ''}caption [[ Image 1 ]]`)
    await settle()
    expect(h.composer().refs.tokensRef.current).toEqual([])

    if (mode !== 'interrupt') {
      const queued = h.composer().refs.queueRef.current[0]
      expect(queued).toMatchObject({ text: 'caption', display: 'caption [[ Image 1 ]]', draftImages: [{ path: '/staged/one.png', label: '[[ Image 1 ]]' }] })
      patchUiState({ busy: false })
      const item = h.composer().actions.dequeue()!
      h.submission().sendQueued(item)
      await settle()
    }

    expect(h.gw.request.mock.calls.filter(([method]) => method === 'session.steer')).toEqual([])
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit').at(-1)).toEqual([
      'prompt.submit', { session_id: 'runtime-a', text: 'caption', draft_image_paths: ['/staged/one.png'] }
    ])
  })

  it('submits only the deliberately admitted draft image paths, then leaves the next turn clean', async () => {
    const h = await mount(true)
    h.gw.stage = Promise.resolve({ path: '/staged/one.png', ref_text: '@file:/staged/one.png', image: { name: 'one.png', mime_type: 'image/png' } })
    h.gw.emit('draft.request', h.request())
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([])
    h.stdin.send('caption', '\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit')).toEqual([
      ['prompt.submit', { session_id: 'runtime-a', text: 'caption', draft_image_paths: ['/staged/one.png'] }]
    ])
    patchUiState({ busy: false })
    h.stdin.send('next turn', '\r')
    await settle()
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'prompt.submit').at(-1)).toEqual([
      'prompt.submit', { session_id: 'runtime-a', text: 'next turn' }
    ])
  })

  it('commits multiple native image tokens locally; removing one never calls global image staging/detach', async () => {
    const h = await mount()

    for (const [id, path] of [['one', '/staged/one.png'], ['two', '/staged/two.png']]) {
      h.gw.stage = Promise.resolve({ path, ref_text: `@file:${path}`, image: { name: `${id}.png`, mime_type: 'image/png' } })
      h.gw.emit('draft.request', h.request(id))
      await settle()
    }

    expect(h.composer().state.input).toBe('[[ Image 1 ]] [[ Image 2 ]]')
    expect(h.composer().refs.tokensRef.current).toEqual([
      expect.objectContaining({ kind: 'image', label: '[[ Image 1 ]]', path: '/staged/one.png', source: 'draft' }),
      expect.objectContaining({ kind: 'image', label: '[[ Image 2 ]]', path: '/staged/two.png', source: 'draft' })
    ])
    h.stdin.send('\x01', '\x0b')
    await settle()
    expect(h.composer().refs.tokensRef.current).toEqual([])
    expect(h.gw.request.mock.calls.filter(([method]) => method.startsWith('image.'))).toEqual([])
    expect(h.submits).toEqual([])
  })

  it('coalesces duplicate in-flight requests and replays success without reinserting; failed work can retry', async () => {
    const h = await mount()
    const first = deferred(); h.gw.stage = first.promise
    const req = h.request()
    h.gw.emit('draft.request', req); h.gw.emit('draft.request', req)
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'file.attach')).toHaveLength(1)
    first.reject(new Error('offline'))
    await settle()
    expect(h.gw.results.at(-1).status).toBe('failed')
    expect(h.composer().state.input).toBe('')
    h.gw.stage = Promise.resolve({ path: '/staged/a', ref_text: '@file:/staged/a' })
    h.gw.emit('draft.request', req)
    await settle()
    const resultsBeforeReplay = h.gw.results.length
    h.gw.emit('draft.request', { path: req.path, expected: { connection_generation: 1, draft_id: req.expected.draft_id, session_id: req.expected.session_id, pty_instance: 'pty-a' }, request_id: req.request_id, type: req.type })
    await settle()
    expect(h.gw.results).toHaveLength(resultsBeforeReplay + 1)
    expect(h.gw.request.mock.calls.filter(([method]) => method === 'file.attach')).toHaveLength(2)
    expect(h.composer().state.input).toBe('@file:/staged/a ')
    expect(h.gw.results.at(-1).status).toBe('attached')
  })

  it.each(['clear', 'submit', 'replacement', 'session', 'overlay', 'disconnect'])('retires an async stage synchronously on %s', async reason => {
    const h = await mount()
    const stage = deferred(); h.gw.stage = stage.promise
    const req = h.request()
    h.stdin.send('old draft')
    h.gw.emit('draft.request', req)

    const replacements: Record<string, () => void> = {
      clear: () => h.composer().actions.clearIn(),
      submit: () => h.stdin.send('\r'),
      replacement: () => h.composer().actions.setInput('replacement'),
      session: () => { patchUiState({ sid: 'runtime-b' }); patchUiState({ sid: 'runtime-a' }) },
      overlay: () => { patchOverlayState({ modelPicker: true }); patchOverlayState({ modelPicker: false }) },
      disconnect: () => h.gw.emit('draft.disconnected')
    }

    replacements[reason]!()
    const next = h.request('next')
    expect(next.expected.draft_id).not.toBe(req.expected.draft_id)
    stage.resolve({ path: '/staged/notes.pdf', ref_text: '@file:/staged/notes.pdf' })
    await settle()
    expect(h.gw.results).toEqual([expect.objectContaining({ status: 'stale' })])
    expect(h.composer().state.input).not.toContain('@file:')
    expect(h.composer().refs.tokensRef.current).toEqual([])
  })

  it('stages a file and acknowledges the live caret insertion without submitting or replacing edits', async () => {
    const h = await mount()
    const stage = deferred(); h.gw.stage = stage.promise
    const req = h.request()
    h.stdin.send('before after')
    h.gw.emit('draft.request', req)
    await settle()
    h.stdin.send('\x1b[D', '\x1b[D', '\x1b[D', '\x1b[D', '\x1b[D', 'new ')
    stage.resolve({ path: '/staged/notes.pdf', ref_text: '@file:/staged/notes.pdf', name: 'notes.pdf' })
    await settle()
    expect(h.composer().state.input).toBe('before new @file:/staged/notes.pdf after')
    expect(h.gw.results).toEqual([expect.objectContaining({ request_id: req.request_id, status: 'attached', identity: req.expected, label: '@file:/staged/notes.pdf' })])
    expect(h.gw.request.mock.calls.filter(([method]) => method.includes('attach')).map(([method]) => method)).toEqual(['file.attach'])
    expect(h.submits).toEqual([])
    h.stdin.send('typed ')
    await settle()
    expect(h.composer().state.input).toBe('before new @file:/staged/notes.pdf typed after')
  })
})
