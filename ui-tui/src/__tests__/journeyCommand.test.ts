import { beforeEach, describe, expect, it } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { findSlashCommand } from '../app/slash/registry.js'

describe('/journey slash command', () => {
  beforeEach(() => {
    resetOverlayState()
  })

  it('resolves by name and aliases', () => {
    expect(findSlashCommand('journey')?.name).toBe('journey')

    for (const alias of ['learning', 'memory-graph']) {
      expect(findSlashCommand(alias)?.name).toBe('journey')
    }
  })

  it('opens the journey overlay when run', () => {
    expect(getOverlayState().journey).toBe(false)
    findSlashCommand('journey')!.run('', {} as never, 'journey')
    expect(getOverlayState().journey).toBe(true)
  })

  it('is preserved by the flow-overlay soft reset (deliberate, user-opened)', async () => {
    findSlashCommand('journey')!.run('', {} as never, 'journey')
    // Mirror turnController.idle(): flow overlays clear, user-opened panels stay.
    const { resetFlowOverlays } = await import('../app/overlayStore.js')
    resetFlowOverlays()
    expect(getOverlayState().journey).toBe(true)
  })
})
