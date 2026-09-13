import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'
import { useQueue } from '../hooks/useQueue.js'

it('submits interpolated queued input only to its captured owner and removes it only after the matching receipt', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-submit-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  const info = { model: 'test', profile_name: 'alpha', stored_session_id: 'stored-owner', skills: {}, tools: {} }
  patchUiState({ sid: 'owner', info })
  let resolveShell!: (value: unknown) => void
  let resolveSubmit!: (value: unknown) => void

  const request = vi.fn((method: string) => {
    if (method === 'shell.exec') {
      return new Promise(resolve => {
        resolveShell = resolve
      })
    }

    if (method === 'input.detect_drop') {
      return Promise.resolve({ matched: false })
    }

    return new Promise(resolve => {
      resolveSubmit = resolve
    })
  })

  let queue!: ReturnType<typeof useQueue>
  let submission!: ReturnType<typeof useSubmission>
  const appended = vi.fn()

  function Harness() {
    queue = useQueue()
    submission = useSubmission({
      appendMessage: appended,
      composerActions: { ...queue, prependQueue: queue.prependQ, takeQueue: queue.takeQ } as any,
      composerRefs: { queueRef: queue.queueRef, queueEditRef: queue.queueEditRef, tokensRef: { current: [] } } as any,
      composerState: { input: '', inputBuf: [], completions: [] } as any,
      gw: { request } as any,
      setLastUserMsg: vi.fn(),
      slashRef: { current: () => true },
      submitRef: { current: () => {} },
      sys: vi.fn()
    })

    return <Text>{queue.queuedDisplay.join('|')}</Text>
  }

  const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false })

  const instance = renderSync(<Harness />, {
    stdin: new PassThrough() as any,
    stdout: stdout as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  })

  try {
    queue.enqueue('private {!printf data}')
    const item = queue.dequeue()!
    submission.sendQueued(item)
    patchUiState({ sid: 'other', busy: false })
    resolveShell({ stdout: 'data', code: 0 })
    await expect.poll(() => request.mock.calls.filter(call => call[0] === 'prompt.submit').length).toBe(1)
    expect(request).toHaveBeenCalledWith('prompt.submit', {
      session_id: 'owner',
      text: 'private data',
      submission_id: item.submissionId,
      queued: true
    })
    expect(appended).not.toHaveBeenCalled()
    expect(queue.queueRef.current).toEqual([])
    patchUiState({ sid: 'owner' })
    expect(queue.queueRef.current).toContain(item)
    resolveSubmit({
      admission_id: item.submissionId,
      target_session_id: 'stored-owner',
      target_profile_home: home,
      status: 'queued'
    })
    await expect.poll(() => queue.queueRef.current.length).toBe(0)
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})
