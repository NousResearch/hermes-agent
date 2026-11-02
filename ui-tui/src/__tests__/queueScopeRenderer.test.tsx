import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useQueue } from '../hooks/useQueue.js'

const info = (profile_name: string) => ({ model: 'test', profile_name, skills: {}, tools: {} })

afterEach(resetUiState)

describe('pending input destination', () => {
  it.each([
    { profile: 'alpha', sid: 'other-session' },
    { profile: 'beta', sid: 'same-session' }
  ])('does not drain or edit another destination: $profile/$sid', async destination => {
    const home = mkdtempSync(join(tmpdir(), 'ink-pending-'))
    vi.stubEnv('HERMES_HOME', home)
    patchUiState({ info: null, sid: null })
    let queue!: ReturnType<typeof useQueue>

    function Harness() {
      queue = useQueue()

      return <Text>{queue.queuedDisplay.join('|') || 'empty queue'}</Text>
    }

    const stdout = new PassThrough()
    Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
    let output = ''
    stdout.on('data', chunk => {
      output += String(chunk)
    })

    const renderOptions = {
      patchConsole: false,
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream
    }

    let instance = renderSync(<Harness />, renderOptions)

    try {
      queue.enqueue('startup payload')
      patchUiState({ info: info('alpha'), sid: 'same-session' })
      const pending = queue.dequeue()!
      expect(pending).toMatchObject({ text: 'startup payload', submissionId: expect.any(String) })
      expect(queue.queueRef.current).toContain(pending)
      expect(queue.dequeue()).toBeUndefined()
      pending.settle!(false)
      expect(queue.dequeue()).toBeUndefined() // no automatic ambiguous replay
      instance.unmount()
      instance = renderSync(<Harness />, renderOptions)
      expect(queue.dequeue()).toBeUndefined()
      const restored = queue.dequeue(true)!
      expect(restored).toMatchObject({ submissionId: pending.submissionId, text: pending.text })
      restored.settle!(true)
      expect(queue.queueRef.current).not.toContain(pending)
      queue.enqueue('private payload', 'private preview')
      queue.setQueueEdit(0)
      patchUiState({ info: info(destination.profile), sid: destination.sid })
      // Input handlers can run before React commits the navigation render.
      expect(queue.dequeue()).toBeUndefined()
      expect(queue.queueRef.current).toEqual([])
      expect(queue.queueEditRef.current).toBeNull()
      queue.enqueue('destination payload')
      await expect.poll(() => queue.queuedDisplay).toEqual(['destination payload'])
      expect(output).not.toContain('private preview|destination payload')
      patchUiState({ info: info('alpha'), sid: 'same-session' })
      expect(queue.queueEditRef.current).toBe(0)
      expect(queue.takeQ(0)).toMatchObject({ display: 'private preview', text: 'private payload' })
      patchUiState({ info: info(destination.profile), sid: destination.sid })
      expect(queue.dequeue()).toMatchObject({ text: 'destination payload' })
      expect(queue.dequeue()).toBeUndefined()
    } finally {
      instance.unmount()
      vi.unstubAllEnvs()
      rmSync(home, { recursive: true, force: true })
    }
  })
})
