import { describe, expect, it } from 'vitest'

import { createSessionQueueManager, removeAtInPlace } from '../hooks/useQueue.js'

describe('removeAtInPlace', () => {
  it('removes the item at the given index in place', () => {
    const arr = ['a', 'b', 'c']

    removeAtInPlace(arr, 1)
    expect(arr).toEqual(['a', 'c'])
  })

  it('is a no-op when the index is out of bounds', () => {
    const arr = ['a', 'b']

    removeAtInPlace(arr, -1)
    removeAtInPlace(arr, 5)
    expect(arr).toEqual(['a', 'b'])
  })

  it('returns the same reference (mutates in place)', () => {
    const arr = ['x']
    const same = removeAtInPlace(arr, 0)

    expect(same).toBe(arr)
    expect(arr).toEqual([])
  })
})

describe('createSessionQueueManager', () => {
  it('keeps queued prompts scoped to their live session', () => {
    const manager = createSessionQueueManager()

    manager.setSession('session-a')
    manager.enqueue('follow-up for A')
    expect(manager.display()).toEqual(['follow-up for A'])

    manager.setSession('session-b')
    expect(manager.display()).toEqual([])

    manager.enqueue('follow-up for B')
    expect(manager.display()).toEqual(['follow-up for B'])

    manager.setSession('session-a')
    expect(manager.display()).toEqual(['follow-up for A'])
    expect(manager.dequeue()).toBe('follow-up for A')
    expect(manager.display()).toEqual([])

    manager.setSession('session-b')
    expect(manager.display()).toEqual(['follow-up for B'])
  })

  it('preserves in-place queue mutations for the active session only', () => {
    const manager = createSessionQueueManager()

    manager.setSession('session-a')
    manager.enqueue('a1')
    manager.setSession('session-b')
    manager.enqueue('b1')

    manager.current().unshift('b0')
    manager.sync()
    expect(manager.display()).toEqual(['b0', 'b1'])

    manager.setSession('session-a')
    expect(manager.display()).toEqual(['a1'])
  })
})
