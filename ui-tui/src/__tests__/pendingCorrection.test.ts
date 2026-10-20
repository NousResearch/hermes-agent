import { describe, expect, it } from 'vitest'

import { acceptedCorrectionView, steerToHydrate } from '../app/pendingCorrection.js'

describe('acceptedCorrectionView', () => {
  it('parks a canonical steer in the queue strip', () => {
    expect(acceptedCorrectionView('steer', true, 'steer')).toBe('steer-row')
  })

  it('parks a tool-time interrupt in the queue strip — redirect degraded to a pending steer', () => {
    expect(acceptedCorrectionView('interrupt', true, 'steer')).toBe('steer-row')
  })

  it('shows a canonical model redirect as a user row, not a pending steer', () => {
    expect(acceptedCorrectionView('interrupt', true, 'redirect')).toBe('user-row')
  })

  it('still parks a canonical interrupt when an older gateway omits the surface', () => {
    expect(acceptedCorrectionView('interrupt', true, undefined)).toBe('steer-row')
  })

  it('leaves standalone interrupt to the user bubble the owned turn already painted', () => {
    expect(acceptedCorrectionView('interrupt', false, undefined)).toBe('none')
    expect(acceptedCorrectionView('interrupt', false, 'redirect')).toBe('none')
  })

  it('parks a standalone steer', () => {
    expect(acceptedCorrectionView('steer', false, undefined)).toBe('steer-row')
  })
})

describe('steerToHydrate', () => {
  it('restores a pending steer after resume when the composer has none', () => {
    expect(steerToHydrate(null, { steer: 'add the marker' })).toBe('add the marker')
  })

  it('does not clobber a steer the user just accepted', () => {
    expect(steerToHydrate('local', { steer: null })).toBeNull()
    expect(steerToHydrate('local', { steer: 'server' })).toBeNull()
  })

  it('ignores an empty snapshot', () => {
    expect(steerToHydrate(null, { steer: null })).toBeNull()
    expect(steerToHydrate(null, null)).toBeNull()
  })
})
