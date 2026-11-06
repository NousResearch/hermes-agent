import { randomUUID } from 'node:crypto'
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import type { ComposerToken } from '../app/interfaces.js'
import { parseQueueCommand } from '../app/queueCommands.js'
import { captureDestination } from '../app/submissionDestination.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'
import { useQueue } from '../hooks/useQueue.js'
import * as journal from '../lib/pendingInputs.js'

const scratch = join(tmpdir(), 'hermes-opsx-apply-20261105', 'finish-queue-slash')
const cleanups: Array<() => void> = []
const noop = () => {}
const flush = () => new Promise(resolve => setImmediate(resolve))

afterEach(() => {
  while (cleanups.length) { cleanups.pop()!() }
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
  turnController.fullReset()
  resetUiState()
})

function mount() {
  mkdirSync(scratch, { recursive: true })
  const home = realpathSync(mkdtempSync(join(scratch, 'mounted-')))
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('TMPDIR', home)
  resetUiState()
  patchUiState({ sid: 'owner', busy: true, gatewayConnected: true,
    info: { model: 'test', skills: {}, tools: {}, profile_name: 'default', stored_session_id: 'stored-owner' } as any })
  const destination = captureDestination()
  const foreign = { ...destination, sid: 'foreign', storedSid: 'stored-foreign' }
  const otherProfile = { ...destination, profile: 'other', profileHome: join(home, 'other'), ownerHome: join(home, 'other') }
  const foreignItem = { submissionId: randomUUID(), text: 'foreign text', display: 'foreign', destination: foreign }
  const profileItem = { submissionId: randomUUID(), text: 'profile text', display: 'profile', destination: otherProfile }
  journal.savePendingInput(foreignItem)
  journal.savePendingInput(profileItem)
  const sys = vi.fn()
  const appendMessage = vi.fn()
  const pushHistory = vi.fn()
  const clearIn = vi.fn()
  const enqueue = vi.fn((...args: Parameters<ReturnType<typeof useQueue>['enqueue']>) => queue.enqueue(...args))
  const remove = vi.fn((index: number) => queue.removeQ(index))
  const gw = { isCanonical: true, request: vi.fn(() => Promise.resolve({})) } as any
  const tokensRef = { current: [] as ComposerToken[] }
  const slashRef = { current: vi.fn() as any }
  let queue!: ReturnType<typeof useQueue>
  let submission!: ReturnType<typeof useSubmission>
  let instance: ReturnType<typeof renderSync>
  function Harness() {
    queue = useQueue(gw)
    slashRef.current = vi.fn(createSlashHandler({
      gateway: { gw }, local: { catalog: null }, slashFlightRef: { current: 0 },
      composer: { queueRef: queue.queueRef, enqueue, removeQueue: remove },
      transcript: { sys }
    } as any))
    submission = useSubmission({
      appendMessage, composerActions: { enqueue, pushHistory, clearIn, removeQueue: remove } as any,
      composerRefs: { queueRef: queue.queueRef, queueEditRef: queue.queueEditRef, tokensRef } as any,
      composerState: { input: '', inputBuf: [], completions: [] } as any,
      gw, setLastUserMsg: noop, slashRef, submitRef: { current: noop }, sys
    })
    return <Text>{queue.queuedDisplay.join('|')}</Text>
  }
  const render = () => renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 30, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })
  instance = render()
  cleanups.push(() => { instance.unmount(); instance.cleanup(); rmSync(home, { recursive: true, force: true }) })
  return {
    home, destination, foreign, otherProfile, foreignItem, profileItem, gw, sys, appendMessage,
    pushHistory, clearIn, enqueue, remove, tokensRef,
    get queue() { return queue }, get slash() { return slashRef.current },
    get output() { return sys.mock.calls.map(([text]) => text).join('\n') },
    seed(...texts: string[]) { texts.forEach(text => queue.enqueue(text, `visible ${text}`, destination)); enqueue.mockClear() },
    dispatch(text: string) { submission.dispatchSubmission(text) },
    pending() { return journal.loadPendingInputs(destination) },
    remount() { instance.unmount(); instance.cleanup(); instance = render() },
    receipt(index: number, status = 'queued', overrides = {}) {
      const item = queue.queueRef.current[index]!
      const receipt = { admission_id: `adm-${item.submissionId}`, input_id: item.submissionId,
        target_session_id: destination.storedSid!, target_profile_home: home, text: item.text, status, ...overrides }
      patchUiState({ info: { ...getUiState().info!, pending_submissions: [receipt] } as any })
      return receipt
    },
    assertNoPromptRpc() {
      expect(gw.request.mock.calls.filter(([method]: [string]) => method !== 'shared_metrics.slash_command')).toEqual([])
    },
    assertForeign() {
      expect(journal.loadPendingInputs(foreign)).toEqual([expect.objectContaining(foreignItem)])
      expect(journal.loadPendingInputs(otherProfile)).toEqual([expect.objectContaining(profileItem)])
    }
  }
}

describe('pure queue command grammar', () => {
  it.each(['', ' ', 'list', '\tlist\n'])('lists %j', args => expect(parseQueueCommand(args)).toEqual({ kind: 'list' }))
  it.each(['rm 1', 'remove 2', 'rm 9007199254740991'])('accepts positive safe decimal %j', args => {
    expect(parseQueueCommand(args)).toEqual({ kind: 'remove', index: Number(args.split(' ')[1]) })
  })
  it.each(['list extra', 'clear extra', 'rm', 'remove', 'rm 0', 'rm -1', 'rm 1.5', 'rm 1e2',
    'rm 9007199254740992', 'rm 1abc', 'rm +1', 'rm 01', 'remove 1 extra'])('rejects %j without prompt fallback', args => {
    expect(parseQueueCommand(args)).toEqual({ kind: 'error' })
  })
  it.each(['List', 'CLEAR', 'say clear', 'ordinary\nmultiline', '[[ paste ]]'])('preserves prompt %j', args => {
    expect(parseQueueCommand(args)).toEqual({ kind: 'prompt' })
  })
})

describe.each(['queue', 'q'])('real /%s submission and local administration', alias => {
  it.each(['', 'list'])('lists local rows, status and stable indices for %j exactly once', async arg => {
    const h = mount()
    h.seed('first', 'second')
    h.queue.queueRef.current[0]!.failed = true
    h.queue.queueRef.current[1]!.inFlight = true
    h.dispatch(`/${alias}${arg ? ` ${arg}` : ''}`)
    expect(h.output).toContain('LOCAL queue')
    expect(h.output).toContain('1. [unconfirmed] visible first')
    expect(h.output).toContain('2. [in flight — protected] visible second')
    expect(h.slash).toHaveBeenCalledTimes(1)
    expect(h.appendMessage).toHaveBeenCalledTimes(1)
    expect(h.pushHistory).toHaveBeenCalledTimes(1)
    expect(h.clearIn).toHaveBeenCalledTimes(1)
    expect(h.enqueue).not.toHaveBeenCalled()
    expect(h.remove).not.toHaveBeenCalled()
    expect(h.gw.request).toHaveBeenCalledTimes(1)
    h.assertNoPromptRpc()
    await flush()
  })

  it('reports an empty LOCAL queue without enqueue or RPC', () => {
    const h = mount()
    h.dispatch(`/${alias}`)
    expect(h.output).toContain('LOCAL queue is empty')
    expect(h.enqueue).not.toHaveBeenCalled()
    h.assertNoPromptRpc()
  })

  it.each(['rm 2', 'remove 2'])('durably removes the local 1-based index via %s and survives real remount', arg => {
    const h = mount()
    h.seed('first', 'second', 'third')
    h.dispatch(`/${alias} ${arg}`)
    expect(h.remove).toHaveBeenCalledExactlyOnceWith(1)
    expect(h.pending().map(item => item.text)).toEqual(['first', 'third'])
    expect(h.output).toContain('removed LOCAL queue item 2')
    h.remount()
    expect(h.queue.queueRef.current.map(item => item.text)).toEqual(['first', 'third'])
    h.assertForeign()
    h.assertNoPromptRpc()
  })

  it.each(['list extra', 'clear extra', 'rm', 'remove', 'rm 0', 'rm -1', 'rm 1.5', 'rm 1e2',
    'rm 9007199254740992', 'rm 1abc', 'remove 1 extra', 'rm 3'])('rejects invalid administrative %j at dispatchSubmission without journal or RPC', arg => {
    const h = mount()
    h.seed('first', 'second')
    const before = h.pending()
    const save = vi.spyOn(journal, 'savePendingInput')
    const remove = vi.spyOn(journal, 'removePendingInput')
    h.dispatch(`/${alias} ${arg}`)
    expect(h.enqueue).not.toHaveBeenCalled()
    expect(h.remove).not.toHaveBeenCalled()
    expect(save).not.toHaveBeenCalled()
    expect(remove).not.toHaveBeenCalled()
    expect(h.pending()).toEqual(before)
    expect(h.output).toMatch(/usage:|invalid LOCAL/)
    h.assertNoPromptRpc()
  })

  it('clears synchronously in descending local order, preserving in-flight and foreign journals', () => {
    const h = mount()
    h.seed('first', 'protected', 'third')
    h.queue.queueRef.current[1]!.inFlight = true
    h.dispatch(`/${alias} clear`)
    expect(h.remove.mock.calls).toEqual([[2], [0]])
    expect(h.pending().map(item => item.text)).toEqual(['protected'])
    expect(h.output).toContain('2 removed, 1 retained')
    h.remount()
    expect(h.queue.queueRef.current.map(item => item.text)).toEqual(['protected'])
    h.assertForeign()
    h.assertNoPromptRpc()
  })

  it.each(['queued', 'unknown', 'started'])('protects matching %s authority receipts before reconciliation at core and removeQ boundaries', status => {
    const h = mount()
    h.seed('protected', 'eligible')
    const receipt = h.receipt(0, status)
    const saved = h.pending()
    h.queue.removeQ(0)
    expect(h.pending()).toEqual(saved)
    h.dispatch(`/${alias} rm 1`)
    expect(h.remove).not.toHaveBeenCalled()
    expect(h.output).toContain('authority-owned')
    h.dispatch(`/${alias} list`)
    expect(h.output).toContain('1. [authority-owned — protected]')
    h.dispatch(`/${alias} clear`)
    expect(h.remove.mock.calls).toEqual([[1]])
    expect(h.pending().map(item => item.text)).toEqual(['protected'])
    expect(getUiState().info!.pending_submissions).toEqual([receipt])
    h.assertNoPromptRpc()
  })

  it('rejects server-tail indices and never cancels queued, unknown or started authority work', async () => {
    const h = mount()
    h.seed('local')
    const receipts = ['queued', 'unknown', 'started'].map((status, i) => ({ admission_id: `adm-${i}`,
      input_id: `server-${i}`, target_session_id: 'stored-owner', target_profile_home: h.home,
      user: `server ${status}`, status }))
    patchUiState({ info: { ...getUiState().info!, pending_submissions: receipts } as any })
    await flush()
    h.dispatch(`/${alias} rm 2`)
    expect(h.remove).not.toHaveBeenCalled()
    expect(h.output).toContain('invalid LOCAL queue index')
    h.dispatch(`/${alias} clear`)
    expect(h.pending()).toEqual([])
    expect(getUiState().info!.pending_submissions).toEqual(receipts)
    h.remount()
    expect(h.queue.queueRef.current).toEqual([])
    expect(h.queue.queuedDisplay.join('|')).toContain('server queued')
    h.assertNoPromptRpc()
  })

  it('retains a failed rm journal write across remount and reports no success', () => {
    const h = mount()
    h.seed('first', 'second')
    vi.spyOn(journal, 'removePendingInput').mockImplementationOnce(() => { throw new Error('disk full') })
    h.dispatch(`/${alias} rm 1`)
    expect(h.output).toContain('removal not saved; item retained')
    expect(h.output).not.toContain('removed LOCAL queue item')
    h.remount()
    expect(h.queue.queueRef.current.map(item => item.text)).toEqual(['first', 'second'])
    h.assertNoPromptRpc()
  })

  it('reports durable partial clear and stops after a journal failure with remaining work retained on reload', () => {
    const h = mount()
    h.seed('first', 'second', 'third')
    const original = journal.removePendingInput
    vi.spyOn(journal, 'removePendingInput').mockImplementationOnce(original)
      .mockImplementationOnce(() => { throw new Error('disk full') })
    h.dispatch(`/${alias} clear`)
    expect(h.remove.mock.calls).toEqual([[2], [1]])
    expect(h.output).toContain('1 removed, 2 retained')
    expect(h.output).toContain('disk full')
    h.remount()
    expect(h.queue.queueRef.current.map(item => item.text)).toEqual(['first', 'second'])
    h.assertForeign()
    h.assertNoPromptRpc()
  })

  it('never claims a void removeQueue no-op removed an item', () => {
    const h = mount()
    h.seed('first')
    h.remove.mockImplementation(() => {})
    h.dispatch(`/${alias} rm 1`)
    expect(h.output).toContain('retained; removal was not confirmed')
    h.dispatch(`/${alias} clear`)
    expect(h.output).toContain('0 removed, 1 retained')
    expect(h.pending()).toHaveLength(1)
  })

  it('stops a partial clear when the destination changes synchronously', () => {
    const h = mount()
    h.seed('first', 'second')
    h.remove.mockImplementation(index => { h.queue.removeQ(index); patchUiState({ sid: 'foreign' }) })
    h.dispatch(`/${alias} clear`)
    expect(h.remove.mock.calls).toEqual([[1]])
    expect(h.output).toContain('operation stopped')
    expect(h.output).toContain('1 removed, 1 retained')
    h.assertForeign()
  })

  it.each(['ordinary prompt', 'multiline\nprompt', 'List', 'CLEAR', 'say clear'])('preserves one journaled enqueue for %j', prompt => {
    const h = mount()
    h.dispatch(`/${alias} ${prompt}`)
    expect(h.enqueue).toHaveBeenCalledExactlyOnceWith(prompt, prompt, h.destination, [])
    expect(h.slash).not.toHaveBeenCalled()
    expect(h.pending()).toEqual([expect.objectContaining({ text: prompt, display: prompt, destination: h.destination })])
    expect(h.appendMessage).toHaveBeenCalledTimes(1)
    expect(h.pushHistory).toHaveBeenCalledTimes(1)
    h.assertNoPromptRpc()
  })

  it('does not reparse expanded administrative words and retains display, attachments and captured destination', () => {
    const h = mount()
    const paste = { kind: 'paste' as const, label: '[[ paste ]]', text: 'clear' }
    const image = { kind: 'image' as const, label: '[[ Image 1 ]]', path: '/captured/image.png', mime: 'image/png', index: 1 }
    h.tokensRef.current = [paste, image]
    h.dispatch(`/${alias} ${paste.label} ${image.label}`)
    expect(h.enqueue).toHaveBeenCalledTimes(1)
    expect(h.slash).not.toHaveBeenCalled()
    expect(h.pending()[0]).toMatchObject({ text: 'clear [[ Image 1 ]]', display: '[[ paste ]] [[ Image 1 ]]', destination: h.destination,
      attachments: [{ path: image.path, mime: image.mime }] })
    h.assertNoPromptRpc()
  })

  it('enqueues a paste expanding to an exact reserved verb without reparsing', () => {
    const h = mount()
    h.seed('existing')
    h.tokensRef.current = [{ kind: 'paste', label: '[[ paste ]]', text: 'clear' }]
    h.dispatch(`/${alias} [[ paste ]]`)
    expect(h.slash).not.toHaveBeenCalled()
    expect(h.remove).not.toHaveBeenCalled()
    expect(h.pending().map(item => item.text)).toEqual(['existing', 'clear'])
    expect(h.pending()[1]!.display).toBe('[[ paste ]]')
    h.assertNoPromptRpc()
  })

  it.each([{ target_session_id: 'different' }, { target_profile_home: '/different' }, { input_id: 'different' }])(
    'does not protect a nonmatching authority receipt %j', overrides => {
      const h = mount()
      h.seed('eligible')
      h.receipt(0, 'queued', overrides)
      h.dispatch(`/${alias} rm 1`)
      expect(h.pending()).toEqual([])
      expect(h.output).toContain('removed LOCAL queue item 1')
      h.assertNoPromptRpc()
    })

  it('routes raw invalid administration once without expanding tokens into grammar', () => {
    const h = mount()
    h.tokensRef.current = [{ kind: 'paste', label: '[[ paste ]]', text: '1' }]
    const full = `/${alias} rm [[ paste ]]`
    h.dispatch(full)
    expect(h.slash).toHaveBeenCalledTimes(1)
    expect(h.slash.mock.calls[0][0]).toBe(full)
    expect(h.output).toContain('usage: /queue')
    expect(h.enqueue).not.toHaveBeenCalled()
    expect(h.remove).not.toHaveBeenCalled()
    h.assertNoPromptRpc()
  })
})
