import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { submitPrompt } from '../app/submissionCore.js'
import { captureDestination } from '../app/submissionDestination.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { useSessionLifecycle } from '../app/useSessionLifecycle.js'
import { useQueue } from '../hooks/useQueue.js'
import { loadPendingInputs } from '../lib/pendingInputs.js'

async function coldResume(legacy: boolean) {
  const home = mkdtempSync(join(tmpdir(), 'ink-cold-native-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  // A separate interpreter writes the real fsync/rename journal; no shared UI or successor map.
  execFileSync(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', `
    import { randomUUID } from 'node:crypto';
    import { savePendingInput } from './src/lib/pendingInputs.ts';
    const destination = { sid: 'R1', storedSid: 'S', profile: 'default', profileHome: process.env.HERMES_HOME };
    for (const [i, text] of ['ambiguous', 'waiting'].entries()) savePendingInput({
      submissionId: randomUUID(), destination, text, display: text, createdAt: i,
      inFlight: i === 0, preparedText: i === 0 ? 'frozen payload' : undefined,
      legacyAttempted: i === 0 && ${legacy}
    });
    savePendingInput({ submissionId: randomUUID(), destination: { ...destination, profile: 'foreign' }, text: 'foreign', display: 'foreign' });
    savePendingInput({ submissionId: randomUUID(), destination: { ...destination, storedSid: 'other' }, text: 'other', display: 'other' });
  `], { cwd: process.cwd(), env: { ...process.env, HERMES_HOME: home }, stdio: 'pipe' })
  let queue!: ReturnType<typeof useQueue>
  let lifecycle!: ReturnType<typeof useSessionLifecycle>

  const request = vi.fn(async (method: string, params: any) => method === 'session.resume'
    ? { session_id: 'R2', session_key: 'S', messages: [], running: false, info: { model: 'test', tools: {}, skills: {}, stored_session_id: 'S', profile_name: 'default', running: false } }
    : { admission_id: params.submission_id, target_session_id: 'S', target_profile_home: home, status: 'queued' })

  function Harness() {
    queue = useQueue()
    lifecycle = useSessionLifecycle({ colsRef: { current: 80 }, composerActions: { setComposerTokens: vi.fn() },
      gw: { request }, rpc: async () => ({}), scrollRef: { current: null }, panel: vi.fn(), sys: vi.fn(),
      setHistoryItems: vi.fn(), setLastUserMsg: vi.fn(), setSessionStartedAt: vi.fn(), setStickyPrompt: vi.fn(),
      setVoiceProcessing: vi.fn(), setVoiceRecording: vi.fn() } as any)

    return <Text>{queue.queuedDisplay.join('|')}</Text>
  }

  const instance = renderSync(<Harness />, { stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any, patchConsole: false })

  try {
    lifecycle.resumeById('S')
    await vi.waitFor(() => expect(getUiState().sid).toBe('R2'))
    expect(queue.queueRef.current.map(item => item.text)).toEqual(['ambiguous', 'waiting'])
    expect(queue.dequeue()).toBeUndefined()
    const attempted = queue.queueRef.current[0]!
    expect(attempted.destination).toMatchObject({ sid: 'R1', storedSid: 'S' })
    expect(queue.queueRef.current[1]!.destination).toMatchObject({ sid: 'R2', storedSid: 'S' })
    expect(loadPendingInputs({ ...captureDestination(), profileHome: join(home, 'different-profile') })).toEqual([])
    const item = queue.dequeue(true)!
    submitPrompt(item.text, { gw: { request } as any, appendMessage: vi.fn(), enqueue: vi.fn(), expand: vi.fn(), setLastUserMsg: vi.fn(), sys: vi.fn() }, false, undefined,
      { destination: item.destination, queueItem: item, skipDetectDrop: true })
    await new Promise(resolve => setImmediate(resolve))
    const submissions = request.mock.calls.filter(([method]) => method === 'prompt.submit')

    if (legacy) {
      expect(submissions).toEqual([])
      expect(queue.queueRef.current[0]).toMatchObject({ failed: true, legacyAttempted: true })
    } else {
      expect(submissions).toEqual([['prompt.submit', { session_id: 'R2', submission_id: item.submissionId, text: 'frozen payload', queued: true }]])
      expect(item.destination).toMatchObject({ sid: 'R1', storedSid: 'S' })
      expect(queue.queueRef.current.map(pending => pending.text)).toEqual(['waiting'])
      expect(queue.dequeue()!.destination?.sid).toBe('R2')
    }
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
}

it('cold-resumes native queued and ambiguous records by durable identity with a live retry target', async () => coldResume(false))
it('cold recovery excludes foreign identities and never replays an ambiguous legacy attempt', async () => coldResume(true))
