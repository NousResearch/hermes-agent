import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import React from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { $uiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useQueue } from '../hooks/useQueue.js'

const info = (profile_name: string) => ({ model: 'test', profile_name, skills: {}, tools: {} })

afterEach(resetUiState)

describe('queue scope subscription', () => {
  it('ignores unrelated ui-state churn but wakes on destination changes', async () => {
    patchUiState({ info: info('alpha'), sid: 'session-a' })
    let renders = 0

    function Harness() {
      renders += 1
      useQueue()

      return <Text>queue probe</Text>
    }

    const stdout = new PassThrough()
    Object.assign(stdout, { columns: 80, isTTY: false, rows: 20 })
    const instance = renderSync(<Harness />, {
      patchConsole: false,
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream
    })

    try {
      const baseline = renders

      for (let i = 0; i < 20; i++) {
        patchUiState({ status: `streaming-${i}` })
      }

      await new Promise(resolve => setTimeout(resolve, 0))
      expect(renders).toBe(baseline)

      patchUiState({ sid: 'session-b' })
      await expect.poll(() => renders).toBe(baseline + 1)

      patchUiState({ info: info('beta') })
      await expect.poll(() => renders).toBe(baseline + 2)
    } finally {
      instance.unmount()
    }
  })
  it('preserves queue display identity through full-state parent renders', async () => {
    patchUiState({ info: info('alpha'), sid: 'session-a' })
    let queue!: ReturnType<typeof useQueue>
    let renders = 0

    // Production shape: useMainApp subscribes to the entire UI state above
    // useComposerState -> useQueue; narrowing useQueue alone cannot skip renders.
    function MainAppShape() {
      useStore($uiState)
      queue = useQueue()
      renders += 1

      return <Text>{queue.queuedDisplay.join('|') || 'empty'}</Text>
    }

    const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 20 })
    const instance = renderSync(<MainAppShape />, {
      patchConsole: false,
      stdin: new PassThrough() as unknown as NodeJS.ReadStream,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream
    })

    try {
      const initialDisplay = queue.queuedDisplay
      const initialRenders = renders

      // Separate ticks avoid React batching hiding the real parent rerenders.
      for (let i = 0; i < 8; i++) {
        patchUiState({ status: `streaming-${i}` })
        await new Promise(resolve => setTimeout(resolve, 0))
      }

      expect(renders).toBeGreaterThan(initialRenders)
      expect(queue.queuedDisplay).toBe(initialDisplay)

      queue.enqueue('hello', 'hello preview')
      await expect.poll(() => queue.queuedDisplay).toEqual(['hello preview'])
      expect(queue.queuedDisplay).not.toBe(initialDisplay)

      const firstSessionDisplay = queue.queuedDisplay
      patchUiState({ sid: 'session-b' })
      await expect.poll(() => queue.queuedDisplay).toEqual([])
      expect(queue.queuedDisplay).not.toBe(firstSessionDisplay)

      patchUiState({ sid: 'session-a' })
      await expect.poll(() => queue.queuedDisplay).toEqual(['hello preview'])
    } finally {
      instance.unmount()
    }
  })

})

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
