import { beforeEach, describe, expect, it, vi } from 'vitest'

import { refreshSlashCatalogOnSkillsChange } from '../app/slashCatalog.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

// CFG-8: a skill installed from another terminal reaches Ink's palette through the gateway.
describe('refreshSlashCatalogOnSkillsChange', () => {
  beforeEach(() => resetUiState())

  it('refetches commands.catalog from the gateway for the live session', async () => {
    const rpc = vi.fn(async () => ({ canon: { '/new-skill': '/new-skill' }, pairs: [['/new-skill', 'd']], skill_count: 1 }))
    const setCatalog = vi.fn()

    patchUiState({ sid: 'sid-cfg8' })

    // SAFETY: the fake answers only commands.catalog, the one method the refetch calls, with that method's result type.
    expect(refreshSlashCatalogOnSkillsChange({ type: 'skills.changed' }, rpc as never, setCatalog)).toBe(true)
    await vi.waitFor(() => expect(setCatalog).toHaveBeenCalled())

    expect(rpc).toHaveBeenCalledWith('commands.catalog', { session_id: 'sid-cfg8' })
    expect(setCatalog.mock.calls.at(-1)?.[0].canon).toEqual({ '/new-skill': '/new-skill' })
  })

  it('leaves every other event to the main handler', () => {
    const rpc = vi.fn()

    // SAFETY: the fake is never called here; the event is not skills.changed, so no RPC goes out.
    expect(refreshSlashCatalogOnSkillsChange({ type: 'skin.changed' }, rpc as never, vi.fn())).toBe(false)
    expect(rpc).not.toHaveBeenCalled()
  })
})
