import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { captureDestination, isCurrentDestination } from '../app/submissionDestination.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useQueue } from '../hooks/useQueue.js'
import * as pending from '../lib/pendingInputs.js'

it('moves compression queues across cached successors without changing attempted receipts or losing settlement', () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-migrate-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'old', info: { model: 'test', profile_name: 'alpha', skills: {}, tools: {} } })
  let queue!: ReturnType<typeof useQueue>

  function Harness() {
    queue = useQueue()

    return <Text>{queue.queuedDisplay.join('|')}</Text>
  }

  const options = {
    stdin: new PassThrough() as any,
    stdout: Object.assign(new PassThrough(), { columns: 80, rows: 20, isTTY: false }) as any,
    stderr: new PassThrough() as any,
    patchConsole: false
  }

  let instance = renderSync(<Harness />, options)

  try {
    const original = captureDestination()
    const attempted = queue.stage('attempted')
    const settle = attempted.settle
    const waiting = queue.enqueue('waiting')
    queue.setQueueEdit(1)
    patchUiState({ sid: 'next' })
    expect(queue.queueRef.current).toEqual([]) // cache successor before migration
    const ownNext = queue.enqueue('already next')
    patchUiState({ sid: 'old' })
    pending.migratePendingInputs(original, 'next')
    patchUiState({ sid: 'next' })
    expect(queue.queueRef.current).toEqual([attempted, waiting, ownNext])
    expect(queue.queueRef.current[0]).toBe(attempted)
    expect(attempted.destination).toEqual(original)
    expect(attempted.settle).toBe(settle)
    expect(waiting.destination?.sid).toBe('next')
    expect(queue.queueEditRef.current).toBe(1)
    expect(pending.loadPendingInputs(original)).toEqual([])
    expect(pending.loadPendingInputs(captureDestination())).toHaveLength(3)
    const next = captureDestination()
    pending.migratePendingInputs(next, 'last')
    patchUiState({ sid: 'last' })
    // Late completion must find the migrated queue even before a getter/render.
    settle!(false)
    expect(queue.queueRef.current[0]).toBe(attempted)
    expect(queue.dequeue()).toBeUndefined()
    instance.unmount()
    instance = renderSync(<Harness />, options)
    const restored = queue.queueRef.current[0]
    expect(restored).toMatchObject({ destination: original, failed: true, submissionId: attempted.submissionId })
    expect(queue.dequeue()).toBeUndefined()
    queue.dequeue(true)!.settle!(true)
    expect(queue.queueRef.current.map(item => item.text)).toEqual(['waiting', 'already next'])
    expect(pending.loadPendingInputs(captureDestination())).toHaveLength(2)
    const live = queue.stage('late success')
    pending.migratePendingInputs(captureDestination(), 'later')
    patchUiState({ sid: 'later' })
    pending.migratePendingInputs(captureDestination(), 'latest')
    patchUiState({ sid: 'latest' })
    live.settle!(true)
    expect(live.destination?.sid).toBe('last')
    expect(queue.queueRef.current.map(item => item.text)).toEqual(['waiting', 'already next'])

    patchUiState({ sid: 'unrelated' })
    expect(queue.queueRef.current).toEqual([])
    queue.enqueue('cold source')
    const cold = captureDestination()
    instance.unmount()
    patchUiState({ sid: 'cold successor' })
    instance = renderSync(<Harness />, options)
    expect(queue.queueRef.current).toEqual([])
    pending.migratePendingInputs(cold, 'cold successor')
    expect(queue.queueRef.current.map(item => item.text)).toEqual(['cold source'])
  } finally {
    instance.unmount()
    resetUiState()
    vi.unstubAllEnvs()
    rmSync(home, { recursive: true, force: true })
  }
})

it('uses the canonical profile home through symlinks, including a missing descendant', () => {
  const root = mkdtempSync(join(tmpdir(), 'ink-home-'))
  const real = join(root, 'real')
  const alias = join(root, 'alias')
  mkdirSync(real)
  symlinkSync(real, alias, 'junction')

  try {
    vi.stubEnv('HERMES_HOME', alias)
    const captured = captureDestination()
    expect(captured.profileHome).toBe(realpathSync(real))
    vi.stubEnv('HERMES_HOME', real)
    expect(isCurrentDestination(captured)).toBe(true)
    vi.stubEnv('HERMES_HOME', join(alias, 'missing', 'nested'))
    expect(captureDestination().profileHome).toBe(join(realpathSync(real), 'missing', 'nested'))
  } finally {
    vi.unstubAllEnvs()
    rmSync(root, { recursive: true, force: true })
  }
})
