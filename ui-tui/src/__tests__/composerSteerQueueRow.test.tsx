import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'

// Steering a queued row is an async RPC: the composer is cleared when it succeeds, but only while
// it still holds the draft the steer started from.
function mount() {
  const home = mkdtempSync(join(tmpdir(), 'ink-steer-queue-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()

  patchUiState({
    busy: true,
    info: {
      pending_submissions: [
        { admission_id: 'adm-1', controls: ['steer', 'replace'], input_id: 'in-1', status: 'queued', user: 'queued text',
          target_profile_home: home, target_session_id: 'stored' }
      ]
    } as any,
    sid: 'owner'
  })

  let finish!: (value: unknown) => void
  const request = vi.fn(() => new Promise(resolve => { finish = resolve }))
  let composer!: ReturnType<typeof useComposerState>

  function Harness() {
    composer = useComposerState({ gw: { request }, submitRef: { current: vi.fn() }, sys: vi.fn() } as any)

    return <Text>{composer.state.input}</Text>
  }

  const instance = renderSync(<Harness />, {
    patchConsole: false,
    stderr: new PassThrough() as any,
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 }) as any
  })

  return {
    composer: () => composer,
    finish: (value: unknown) => finish(value),
    request,
    cleanup() {
      instance.unmount()
      resetUiState()
      vi.unstubAllEnvs()
      rmSync(home, { force: true, recursive: true })
    }
  }
}

afterEach(() => resetUiState())

it('clears the edited row from the composer once the steer lands', async () => {
  const h = mount()

  try {
    h.composer().actions.setQueueEdit(0)
    h.composer().actions.setInput('queued text')

    const steered = h.composer().actions.steerQueue(0)
    h.finish({ outcome: null, status: 'started' })

    expect(await steered).toBe(true)
    expect(h.request).toHaveBeenCalledWith('prompt.steer_queued', { admission_id: 'adm-1', session_id: 'owner' })
    expect(h.composer().refs.queueEditRef.current).toBeNull()
    await expect.poll(() => h.composer().state.input).toBe('')
  } finally {
    h.cleanup()
  }
})

it('keeps text typed while the steer was in flight, and ends the queue edit', async () => {
  const h = mount()

  try {
    h.composer().actions.setQueueEdit(0)
    h.composer().actions.setInput('queued text')

    const steered = h.composer().actions.steerQueue(0)
    h.composer().actions.setInput('queued textnew draft I just typed')
    h.finish({ outcome: null, status: 'started' })

    expect(await steered).toBe(true)
    await expect.poll(() => h.composer().state.input).toBe('queued textnew draft I just typed')
    expect(h.composer().refs.queueEditRef.current).toBeNull()
  } finally {
    h.cleanup()
  }
})

it('leaves the composer and the queue edit alone when the owner refuses the steer', async () => {
  const h = mount()

  try {
    h.composer().actions.setQueueEdit(0)
    h.composer().actions.setInput('queued text')

    const steered = h.composer().actions.steerQueue(0)
    h.finish(Promise.reject(new Error('steer_unavailable')))

    expect(await steered).toBe(false)
    expect(h.composer().refs.queueEditRef.current).toBe(0)
  } finally {
    h.cleanup()
  }
})
