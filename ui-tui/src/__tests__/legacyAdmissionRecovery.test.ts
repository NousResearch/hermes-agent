import { randomUUID } from 'node:crypto'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, it, vi } from 'vitest'

import { submitPrompt, type SubmitPromptDeps } from '../app/submissionCore.js'
import { captureDestination } from '../app/submissionDestination.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { QueueItem } from '../hooks/useQueue.js'
import { loadPendingInputs, savePendingInput } from '../lib/pendingInputs.js'

it('never redispatches an ambiguous legacy attempt after native journal reload', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-legacy-ack-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'runtime', info: { model: 'test', tools: {}, skills: {}, stored_session_id: 'stored' } })
  const destination = captureDestination()

  const item: QueueItem = {
    submissionId: randomUUID(), text: 'exact Ω', display: 'exact Ω', destination, inFlight: true
  }

  item.settle = accepted => {
    item.inFlight = false
    item.failed = !accepted
    savePendingInput(item)
  }

  const request = vi.fn(async (_method: string, params: Record<string, unknown>) => {
    if (params.submission_id) { throw Object.assign(new Error('unsupported'), { code: 4094 }) }
    throw new Error('lost legacy ACK')
  })

  const deps = {
    gw: { request }, appendMessage: vi.fn(), enqueue: vi.fn(), expand: (value: string) => value,
    setLastUserMsg: vi.fn(), sys: vi.fn()
  } as unknown as SubmitPromptDeps

  try {
    submitPrompt(item.text, deps, true, undefined, { destination, queueItem: item, skipDetectDrop: true })
    await vi.waitFor(() => expect(item.failed).toBe(true))
    expect(request).toHaveBeenCalledTimes(2)
    const [restored] = loadPendingInputs(destination)
    expect(restored).toMatchObject({ submissionId: item.submissionId, preparedText: item.text, failed: true })
    restored!.settle = vi.fn()
    submitPrompt(restored!.text, deps, true, undefined, { destination, queueItem: restored, skipDetectDrop: true })
    await new Promise(resolve => setImmediate(resolve))
    expect(request).toHaveBeenCalledTimes(2)
    expect(restored!.settle).toHaveBeenCalledWith(false)
    expect(loadPendingInputs(destination)).toHaveLength(1)
  } finally {
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})
