import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { InputEvent } from '../../packages/hermes-ink/src/ink/events/input-event.js'
import { INITIAL_STATE, parseMultipleKeypresses } from '../../packages/hermes-ink/src/ink/parse-keypress.js'
import type { InputHandlerContext } from '../app/interfaces.js'
import type { QueueItem } from '../hooks/useQueue.js'

// Only the platform choice is controlled; the production platform helpers are
// evaluated below against each actual process.platform branch before import.
const scratch = join(tmpdir(), 'queue-retry-tests')
const cleanups: Array<() => void> = []
const noop = () => {}
const settle = () => new Promise(resolve => setTimeout(resolve, 30))

function decode(sequence: string) {
  const [keys] = parseMultipleKeypresses(INITIAL_STATE, sequence)
  expect(keys).toHaveLength(1)
  const key = keys[0]!
  if (!('name' in key)) { throw new Error('Expected a keyboard event, not mouse input') }
  return new InputEvent(key)
}

async function mount(platform: NodeJS.Platform = 'darwin') {
  // Reload the real helpers under each platform branch, together with Ink so
  // the mounted handlers and renderer retain the same React context instances.
  vi.resetModules()
  const descriptor = Object.getOwnPropertyDescriptor(process, 'platform')!
  let modules
  try {
    Object.defineProperty(process, 'platform', { ...descriptor, value: platform })
    modules = await Promise.all([
      import('@hermes/ink'), import('react'), import('../app/submissionDestination.js'),
      import('../app/turnController.js'), import('../app/uiStore.js'),
      import('../app/useInputHandlers.js'), import('../components/textInput.js'),
      import('../hooks/useQueue.js'), import('../lib/pendingInputs.js')
    ])
  } finally { Object.defineProperty(process, 'platform', descriptor) }
  const [{ renderSync, Text }, { useState }, { captureDestination }, { turnController },
    { patchUiState, resetUiState }, { useInputHandlers }, { TextInput }, { useQueue },
    { loadPendingInputs, savePendingInput }] = modules
  mkdirSync(scratch, { recursive: true })
  const home = realpathSync(mkdtempSync(join(scratch, 'queue-retry-')))
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('TMPDIR', home)
  resetUiState()
  turnController.fullReset()
  patchUiState({ sid: 'owner', gatewayConnected: true, busy: false,
    info: { model: 'test', skills: {}, tools: {}, profile_name: 'default', stored_session_id: 'stored-owner' } as any })

  const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: noop, unref: noop })
  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })
  const dispatchSubmission = vi.fn()
  const setQueueEdit = vi.fn()
  const dequeue = vi.fn((retry?: boolean): QueueItem | undefined => queue.dequeue(retry))
  const changes: string[] = []
  let queue!: ReturnType<typeof useQueue>
  const cursorSnapshotRef = { current: { value: 'prefix suffix', cursor: 7 } }

  function Harness() {
    queue = useQueue()
    const [value, setValue] = useState('prefix suffix')
    useInputHandlers({
      actions: { dispatchSubmission, appendMessage: noop, cancelClarify: noop, die: noop,
        guardBusySessionSwitch: () => false, newSession: noop, sys: noop },
      composer: {
        actions: { dequeue, setQueueEdit, setInput: setValue, openEditor: async () => {}, clearIn: noop },
        refs: { queueRef: queue.queueRef, queueEditRef: queue.queueEditRef, historyRef: { current: [] },
          historyDraftRef: { current: '' }, tokensRef: { current: [] }, submitRef: { current: noop } },
        state: { input: value, inputBuf: [], tokens: [], completions: [], compIdx: 0, compReplace: 0,
          historyIdx: null, queueEditIdx: null, queuedDisplay: queue.queuedDisplay }
      },
      gateway: { gw: { request: vi.fn() }, rpc: async () => null },
      terminal: { stdout, hasSelection: false, scrollRef: { current: null }, scrollWithSelection: noop,
        selection: { clearSelection: noop, copySelection: noop } },
      voice: { enabled: false, recording: false, recordKey: { key: 'b', ctrl: true, alt: false, super: false },
        setProcessing: noop, setRecording: noop, setVoiceEnabled: noop, setVoiceTts: noop },
      wheelStep: 1
    } as unknown as InputHandlerContext)
    return <><Text>{queue.queuedDisplay.join('|')}</Text><TextInput columns={80} cursorSnapshotRef={cursorSnapshotRef}
      onChange={next => { changes.push(next); setValue(next) }} onSubmit={noop} value={value} /></>
  }

  const instance = renderSync(<Harness />, { stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream, stderr: new PassThrough() as any, patchConsole: false })
  cleanups.push(() => { instance.unmount(); instance.cleanup(); turnController.fullReset(); resetUiState();
    vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) })

  return { home, dispatchSubmission, dequeue, setQueueEdit, changes, patchUiState, loadPendingInputs,
    get queue() { return queue },
    async send(sequence: string) { decode(sequence); stdin.write(sequence); await settle() },
    async failedItem() {
      const item = queue.enqueue('original prompt', 'visible prompt', captureDestination(), [{ path: '/image.png', mime: 'image/png' }])!
      item.ownerDestination = captureDestination()
      item.preparedText = 'prepared prompt with image token'
      item.failed = true
      savePendingInput(item)
      await settle()
      return item
    }
  }
}

afterEach(() => { while (cleanups.length) { cleanups.pop()!() }; vi.restoreAllMocks() })

describe.each(['darwin', 'linux', 'win32'] as const)('%s: real decoder, mounted input handlers and composer', platform => {
  it.each([
    ['legacy Alt+K', '\x1bk', false],
    ['legacy Alt+Shift+K', '\x1bK', true],
    ['Kitty Alt+K', '\x1b[107;3u', false],
    ['Kitty Alt+Shift+K', '\x1b[107;4u', true]
  ])('%s retries one local item without changing its identity or prepared payload', async (_label, bytes, shift) => {
    const event = decode(bytes as string)
    expect(event.key).toMatchObject({ meta: true, ctrl: false, super: false, shift })
    const h = await mount(platform)
    const item = await h.failedItem()
    const before = structuredClone(item)
    expect(before.submissionId).toBeTruthy()
    expect(before.destination).toMatchObject({ sid: 'owner', storedSid: 'stored-owner', profileHome: h.home })
    expect(before.ownerDestination).toEqual(before.destination)
    await h.send(bytes as string)
    expect(h.dequeue).toHaveBeenCalledExactlyOnceWith(true)
    expect(h.setQueueEdit).toHaveBeenCalledExactlyOnceWith(null)
    expect(h.dispatchSubmission).toHaveBeenCalledExactlyOnceWith(item)
    expect(h.dequeue.mock.invocationCallOrder[0]).toBeLessThan(h.setQueueEdit.mock.invocationCallOrder[0]!)
    expect(h.setQueueEdit.mock.invocationCallOrder[0]).toBeLessThan(h.dispatchSubmission.mock.invocationCallOrder[0]!)
    const submitted = h.dispatchSubmission.mock.calls[0]![0] as QueueItem
    expect(submitted).toBe(item)
    for (const field of ['submissionId', 'destination', 'ownerDestination', 'text', 'display', 'preparedText', 'attachments'] as const) {
      expect(submitted[field]).toEqual(before[field])
    }
    expect(h.loadPendingInputs(item.destination!)[0]).toMatchObject({ submissionId: before.submissionId, preparedText: before.preparedText })
    await h.send(bytes as string)
    expect(h.dispatchSubmission).toHaveBeenCalledTimes(1)
  })

  it.each([
    ['Ctrl+K', '\x0b', { ctrl: true, meta: false, super: false }],
    ['Kitty Ctrl+Alt+K', '\x1b[107;7u', { ctrl: true, meta: true, super: false }],
    ['Kitty Super+K', '\x1b[107;9u', { ctrl: false, meta: false, super: true }],
    ['Kitty Super+Alt+K', '\x1b[107;11u', { ctrl: false, meta: true, super: true }],
    ['plain k', 'k', { ctrl: false, meta: false, super: false }]
  ])('%s never dequeues or dispatches', async (_label, bytes, modifiers) => {
    expect(decode(bytes as string).key).toMatchObject(modifiers)
    const h = await mount(platform)
    const item = await h.failedItem()
    await h.send(bytes as string)
    expect(h.dequeue).not.toHaveBeenCalled()
    expect(h.dispatchSubmission).not.toHaveBeenCalled()
    expect(h.setQueueEdit).not.toHaveBeenCalled()
    expect(h.queue.queueRef.current).toContain(item)
    if (bytes === '\x0b') { expect(h.changes.at(-1)).toBe('prefix ') }
  })

  it('does not replay server-only queued rows', async () => {
    const h = await mount(platform)
    h.patchUiState({ info: { model: 'test', skills: {}, tools: {}, profile_name: 'default',
      stored_session_id: 'stored-owner', pending_submissions: [{ admission_id: 'server-only',
        input_id: 'server-only', status: 'queued', user: 'already admitted remotely' }] } as any })
    await settle()
    expect(h.queue.queuedDisplay.some(row => row.includes('already admitted remotely'))).toBe(true)
    expect(h.queue.queueRef.current).toHaveLength(0)
    await h.send('\x1bk')
    expect(h.dequeue).not.toHaveBeenCalled()
    expect(h.dispatchSubmission).not.toHaveBeenCalled()
    expect(h.setQueueEdit).not.toHaveBeenCalled()
  })

  it.each(['empty', 'disconnected', 'inFlight', 'no-sid', 'journal-write'] as const)('%s guard prevents dispatch and preserves the pending item', async guard => {
    const h = await mount(platform)
    const item = guard === 'empty' ? undefined : await h.failedItem()
    if (guard === 'disconnected') { h.patchUiState({ gatewayConnected: false }) }
    if (guard === 'inFlight') { item!.inFlight = true }
    if (guard === 'no-sid') { h.patchUiState({ sid: null }) }
    if (guard === 'journal-write') {
      const dir = join(h.home, 'tui-pending-inputs')
      rmSync(dir, { recursive: true, force: true })
      writeFileSync(dir, 'not a directory')
    }
    await settle()
    await h.send('\x1bk')
    expect(h.dispatchSubmission).not.toHaveBeenCalled()
    expect(h.setQueueEdit).not.toHaveBeenCalled()
    if (item && guard !== 'no-sid') { expect(h.queue.queueRef.current).toContain(item) }
    if (guard === 'no-sid') {
      h.patchUiState({ sid: 'owner' })
      await settle()
      expect(h.queue.queueRef.current).toContain(item)
    }
    if (guard === 'journal-write') {
      expect(item!.failed).toBe(true)
      expect(item!.inFlight).toBeUndefined()
    }
  })
})
