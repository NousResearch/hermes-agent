import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useQueue } from '../hooks/useQueue.js'

const info = (profile_name: string) => ({ model: 'test', profile_name, skills: {}, tools: {} })

afterEach(resetUiState)

describe('pending input destination', () => {
  it.each([
    { profile: 'alpha', sid: 'other-session' },
    { profile: 'beta', sid: 'same-session' }
  ])('does not drain or edit another destination: $profile/$sid', async destination => {
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

    const instance = renderSync(<Harness />, {
      patchConsole: false,
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream
    })

    try {
      queue.enqueue('startup payload')
      patchUiState({ info: info('alpha'), sid: 'same-session' })
      expect(queue.dequeue()).toBe('startup payload')
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
      expect(queue.takeQ(0)).toEqual({ display: 'private preview', text: 'private payload' })
      patchUiState({ info: info(destination.profile), sid: destination.sid })
      expect(queue.dequeue()).toBe('destination payload')
      expect(queue.dequeue()).toBeUndefined()
    } finally {
      instance.unmount()
    }
  })
})
