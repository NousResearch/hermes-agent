import { EventEmitter } from 'node:events'
import { mkdirSync, mkdtempSync, realpathSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { captureDestination } from '../app/submissionDestination.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useMainApp } from '../app/useMainApp.js'
import { AppLayout } from '../components/appLayout.js'
import type { QueueItem } from '../hooks/useQueue.js'
import { t } from '../i18n/index.js'
import { loadPendingInputs, savePendingInput } from '../lib/pendingInputs.js'

const scratch = join(tmpdir(), 'queue-head-recovery')
const cleanups: Array<() => void> = []
const flush = () => new Promise(resolve => setTimeout(resolve, 60))
const noop = () => {}

afterEach(() => {
  while (cleanups.length) { cleanups.pop()!() }
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
  turnController.fullReset()
  resetOverlayState()
  resetUiState()
})

async function mount(options: { legacy?: boolean; cols?: number; failed?: boolean } = {}) {
  mkdirSync(scratch, { recursive: true })
  const home = realpathSync(mkdtempSync(join(scratch, 'mounted-')))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  resetOverlayState()
  turnController.fullReset()
  patchUiState({ sid: 'owner', storedSid: 'stored-owner', gatewayConnected: true, busy: false,
    info: { model: 'test', skills: {}, tools: {}, profile_name: 'default', stored_session_id: 'stored-owner' } as any })
  const destination = captureDestination()
  const head: QueueItem = { submissionId: '11111111-1111-4111-8111-111111111111',
    text: 'first original', display: 'first visible', preparedText: 'first prepared', destination,
    ownerDestination: destination, attachments: [{ path: '/owner/image.png', mime: 'image/png' }],
    failed: options.failed ?? true, legacyAttempted: options.legacy }
  const successor: QueueItem = { submissionId: '22222222-2222-4222-8222-222222222222',
    text: 'second original', display: 'second visible', destination, preparedText: 'second prepared' }
  savePendingInput(head)
  savePendingInput(successor)
  const submissions: Array<{ params: any; accept: () => void }> = []
  const gw = Object.assign(new EventEmitter(), {
    attached: true, isCanonical: true, drain: noop, getLogTail: () => '', kill: noop, recordLog: noop,
    publishLocalEvent: noop,
    request: vi.fn((method: string, params: any) => {
      if (method === 'prompt.submit') { return new Promise(resolve => submissions.push({ params,
        accept: () => resolve({ admission_id: `adm-${params.submission_id}`, input_id: params.submission_id,
          status: 'queued', target_session_id: destination.storedSid, target_profile_home: home }) })) }
      if (method === 'input.detect_drop') { return Promise.resolve({ matched: false }) }
      return new Promise(() => {})
    })
  })
  const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: noop, unref: noop, setRawMode: noop })
  const stdout = Object.assign(new PassThrough(), { columns: options.cols ?? 80, rows: 60, isTTY: false })
  let output = ''
  stdout.on('data', chunk => { output += chunk.toString() })
  let app!: ReturnType<typeof useMainApp>
  function Harness() {
    app = useMainApp(gw as never)
    return <GatewayProvider value={app.gateway}><AppLayout actions={app.appActions} composer={app.appComposer}
      mouseTracking="off" progress={app.appProgress} status={app.appStatus} transcript={app.appTranscript} /></GatewayProvider>
  }
  const instance = renderSync(<Harness />, { stdin: stdin as any, stdout: stdout as any,
    stderr: new PassThrough() as any, patchConsole: false })
  cleanups.push(() => { instance.unmount(); instance.cleanup(); rmSync(home, { recursive: true, force: true }) })
  await flush()
  return { home, head, successor, submissions, gw, destination,
    get composer() { return app.appComposer },
    get output() { return output.replace(new RegExp(String.fromCharCode(27) + '\\[[0-9;?]*[A-Za-z]', 'g'), '').replace(/\s+/g, ' ') },
    pending: () => loadPendingInputs(destination),
    async send(bytes: string) { stdin.write(bytes); await flush() },
    async refresh() { patchUiState({ status: `refresh-${Math.random()}` }); await flush() },
    breakJournal() {
      const dir = join(home, 'tui-pending-inputs')
      renameSync(dir, `${dir}.retained`)
      writeFileSync(dir, 'not a directory')
    },
    restoreJournal() {
      const dir = join(home, 'tui-pending-inputs')
      rmSync(dir)
      renameSync(`${dir}.retained`, dir)
    }
  }
}

describe('mounted production queue-head recovery', () => {
  it('keeps the full wrapping recovery line visible at narrow width while another row is selected', async () => {
    const h = await mount({ cols: 42 })
    expect(h.composer.queueRecoveryHint).toBe(t('canonical.queue.blockedRecovery'))
    expect(h.composer.queuedDisplay[0]).toContain(t('canonical.queue.unconfirmedPrefix'))
    expect(h.output).toContain('Queue paused: first input is unconfirmed.')
    expect(h.output).toContain('Ctrl+X discards it.')
    await h.send('\x1b[B')
    expect(h.composer.queueEditIdx).toBe(1)
    expect(h.composer.queueRecoveryHint).toBe(t('canonical.queue.blockedRecovery'))
    expect(h.submissions).toHaveLength(0)
  })

  it('never idle-retries or bypasses the head; explicit Alt+K preserves identity, then acceptance releases FIFO', async () => {
    const h = await mount()
    expect(h.submissions).toHaveLength(0)
    await h.refresh()
    expect(h.submissions).toHaveLength(0)
    await h.send('\x1bk')
    expect(h.submissions).toHaveLength(1)
    expect(h.submissions[0]!.params).toMatchObject({ session_id: 'owner', submission_id: h.head.submissionId,
      text: h.head.preparedText, attachments: h.head.attachments })
    const saved = h.pending()[0]!
    for (const field of ['submissionId', 'destination', 'ownerDestination', 'preparedText', 'attachments', 'text', 'display'] as const) {
      expect(saved[field]).toEqual(h.head[field])
    }
    await h.send('\x1bk')
    expect(h.submissions).toHaveLength(1)
    h.submissions[0]!.accept()
    await flush()
    patchUiState({ busy: false })
    await flush()
    expect(h.submissions).toHaveLength(2)
    expect(h.submissions[1]!.params.submission_id).toBe(h.successor.submissionId)
    expect(h.pending().map(item => item.submissionId)).toEqual([h.successor.submissionId])
    expect(h.composer.queueRecoveryHint).toBeUndefined()
  })

  it('empty composer Up then Ctrl+X durably discards the head and the real idle effect sends its successor', async () => {
    const h = await mount()
    expect(h.composer.input).toBe('')
    await h.send('\x1b[A')
    expect(h.composer.queueEditIdx).toBe(0)
    expect(h.composer.input).toBe(h.head.display)
    await h.send('\x18')
    expect(h.composer.queueEditIdx).toBeNull()
    expect(h.composer.input).toBe('')
    expect(h.pending().map(item => item.submissionId)).toEqual([h.successor.submissionId])
    expect(h.submissions).toHaveLength(1)
    expect(h.submissions[0]!.params.submission_id).toBe(h.successor.submissionId)
  })

  it('failed durable discard retains queue order, composer draft, selection and recovery guidance', async () => {
    const h = await mount()
    await h.send('\x1b[A')
    const before = [...h.composer.queuedDisplay]
    h.breakJournal()
    await h.send('\x18')
    expect(h.composer.queueEditIdx).toBe(0)
    expect(h.composer.input).toBe(h.head.display)
    expect(h.composer.queuedDisplay).toEqual(before)
    expect(h.composer.queueRecoveryHint).toBe(t('canonical.queue.blockedRecovery'))
    expect(getUiState().status).toContain('queue kept')
    expect(h.submissions).toHaveLength(0)
    h.restoreJournal()
    expect(h.pending().map(item => item.submissionId)).toEqual([h.head.submissionId, h.successor.submissionId])
    await h.send('\x18')
    expect(h.composer.queueEditIdx).toBeNull()
    expect(h.composer.input).toBe('')
    expect(h.pending().map(item => item.submissionId)).toEqual([h.successor.submissionId])
    expect(h.submissions[0]!.params.submission_id).toBe(h.successor.submissionId)
  })

  it('failed retry journaling retains the failed head without dispatching or bypassing it', async () => {
    const h = await mount()
    const before = [...h.composer.queuedDisplay]
    h.breakJournal()
    await h.send('\x1bk')
    expect(h.submissions).toHaveLength(0)
    expect(h.composer.queuedDisplay).toEqual(before)
    expect(h.composer.queueRecoveryHint).toBe(t('canonical.queue.blockedRecovery'))
    expect(getUiState().status).toContain('queue kept')
    h.restoreJournal()
    expect(h.pending().map(item => item.submissionId)).toEqual([h.head.submissionId, h.successor.submissionId])
    await h.refresh()
    expect(h.submissions).toHaveLength(0)
  })

  it('legacy uncertainty shows check-session/no-resend copy and never replays on Alt+K', async () => {
    const h = await mount({ legacy: true })
    expect(h.composer.queueRecoveryHint).toBe(t('canonical.submit.deliveryNotResent'))
    expect(h.composer.queueRecoveryHint).not.toContain('retries the same submission')
    await h.send('\x1bk')
    expect(h.submissions).toHaveLength(0)
    expect(h.pending().map(item => item.submissionId)).toEqual([h.head.submissionId, h.successor.submissionId])
  })

  it('does not show local recovery while its submission is in flight', async () => {
    const h = await mount()
    await h.send('\x1bk')
    expect(h.submissions).toHaveLength(1)
    expect(h.composer.queueRecoveryHint).toBeUndefined()
  })

  it.each(['queued', 'unknown', 'started'])('authority %s receipts supersede local recovery', async status => {
    const h = await mount()
    patchUiState({ info: { ...getUiState().info!, pending_submissions: [{ admission_id: 'authority',
      input_id: h.head.submissionId, target_session_id: h.destination.storedSid, target_profile_home: h.home,
      status, user: 'authority head' }] } as any })
    await flush()
    expect(h.composer.queueRecoveryHint).toBeUndefined()
    // The authority receipt confirms the head; the existing idle effect may
    // now submit its successor, but must never replay the confirmed input.
    expect(h.submissions).toHaveLength(1)
    expect(h.submissions[0]!.params.submission_id).toBe(h.successor.submissionId)
    expect(h.pending().map(item => item.submissionId)).toEqual([h.successor.submissionId])
  })

  it('a receipt for a different destination does not hide local recovery', async () => {
    const h = await mount()
    patchUiState({ info: { ...getUiState().info!, pending_submissions: [{ admission_id: 'other',
      input_id: h.head.submissionId, target_session_id: 'different-owner', target_profile_home: h.home,
      status: 'queued', user: 'other head' }] } as any })
    await flush()
    expect(h.composer.queueRecoveryHint).toBe(t('canonical.queue.blockedRecovery'))
    expect(h.submissions).toHaveLength(0)
  })
})
