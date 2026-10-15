import { describe, expect, it } from 'vitest'

import {
  migrateModelSwitchWarnings,
  rememberModelSwitchWarning,
  restoreModelSwitchWarnings
} from '../app/modelSwitchWarnings.js'

describe('model switch warning cache', () => {
  it('migrates warnings when compression rotates the stored session key', () => {
    rememberModelSwitchWarning('before-compression', 'provider changed', 100, 'epoch:1')

    migrateModelSwitchWarnings('before-compression', 'after-compression')

    expect(restoreModelSwitchWarnings('before-compression', [])).toEqual([])
    expect(restoreModelSwitchWarnings('after-compression', [])).toEqual([
      { role: 'system', text: 'warning: provider changed', createdAt: 100 }
    ])
  })

  it('deduplicates an occurrence while merging rotated session keys', () => {
    rememberModelSwitchWarning('rotation-source', 'same warning', 100, 'epoch:2')
    rememberModelSwitchWarning('rotation-target', 'same warning', 100, 'epoch:2')

    migrateModelSwitchWarnings('rotation-source', 'rotation-target')

    expect(restoreModelSwitchWarnings('rotation-target', [])).toHaveLength(1)
  })

  it('keeps the successor newest warning when a rotated session is bounded', () => {
    for (let index = 0; index < 64; index += 1) {
      rememberModelSwitchWarning('full-predecessor', `old-${index}`, index, `old:${index}`)
    }
    rememberModelSwitchWarning('new-successor', 'newest', 1000, 'new:1')

    migrateModelSwitchWarnings('full-predecessor', 'new-successor')

    const restored = restoreModelSwitchWarnings('new-successor', [])
    expect(restored).toHaveLength(64)
    expect(restored.some(row => row.text === 'warning: old-0')).toBe(false)
    expect(restored.some(row => row.text === 'warning: newest')).toBe(true)
  })

  it('bounds retained session keys', () => {
    for (let index = 0; index < 70; index += 1) {
      rememberModelSwitchWarning(`session-${index}`, `warning-${index}`, index, `session-epoch:${index}`)
    }

    expect(restoreModelSwitchWarnings('session-0', [])).toEqual([])
    expect(restoreModelSwitchWarnings('session-69', [])).toHaveLength(1)
  })

  it('bounds retained warnings for a long-lived session', () => {
    for (let index = 0; index < 65; index += 1) {
      rememberModelSwitchWarning('bounded-session', `warning-${index}`, index, `epoch:${index}`)
    }

    const restored = restoreModelSwitchWarnings('bounded-session', [])

    expect(restored).toHaveLength(64)
    expect(restored.some(row => row.text === 'warning: warning-0')).toBe(false)
    expect(restored.some(row => row.text === 'warning: warning-64')).toBe(true)
  })
})
