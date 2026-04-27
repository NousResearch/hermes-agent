import { describe, expect, it } from 'vitest'

import { removeAt } from '../hooks/useQueue.js'

describe('removeAt', () => {
  it('removes the item at the given index in place', () => {
    const arr = ['a', 'b', 'c']

    removeAt(arr, 1)
    expect(arr).toEqual(['a', 'c'])
  })

  it('is a no-op when the index is out of bounds', () => {
    const arr = ['a', 'b']

    removeAt(arr, -1)
    removeAt(arr, 5)
    expect(arr).toEqual(['a', 'b'])
  })

  it('returns the same reference (mutates in place)', () => {
    const arr = ['x']
    const same = removeAt(arr, 0)

    expect(same).toBe(arr)
    expect(arr).toEqual([])
  })
})
