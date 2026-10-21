import { beforeEach, describe, expect, it } from 'vitest'

import type { GatewayRpc } from '../app/interfaces.js'
import { loadSlashCatalog } from '../app/slashCatalog.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { CommandsCatalogResponse } from '../gatewayTypes.js'
import type { SlashCatalog } from '../types.js'

const REPO_SKILL = '/repo-probe'

const catalogFor = (sessionId: string | undefined): CommandsCatalogResponse => {
  const skills = sessionId === 'repo' ? [REPO_SKILL] : []

  return {
    canon: Object.fromEntries(skills.map(skill => [skill, skill])),
    categories: [],
    pairs: skills.map(skill => [skill, 'skill']),
    skill_count: skills.length,
    sub: {}
  }
}

describe('loadSlashCatalog', () => {
  let catalog: null | SlashCatalog

  const setCatalog = (next: null | SlashCatalog) => {
    catalog = next
  }

  beforeEach(() => {
    resetUiState()
    catalog = null
  })

  it("binds the catalog to the live session, so each session's project skills replace the last one's", async () => {
    const requested: (string | undefined)[] = []

    const fake = async (method: string, params: { session_id?: string } = {}) => {
      expect(method).toBe('commands.catalog')
      requested.push(params.session_id)

      return catalogFor(params.session_id)
    }

    // SAFETY: the fake answers only commands.catalog, the one method loadSlashCatalog calls, with that method's result type.
    const rpc = fake as GatewayRpc

    await loadSlashCatalog(rpc, null, setCatalog)
    expect(catalog?.canon[REPO_SKILL]).toBeUndefined()

    patchUiState({ sid: 'repo' })
    await loadSlashCatalog(rpc, 'repo', setCatalog)
    expect(catalog?.canon[REPO_SKILL]).toBe(REPO_SKILL)
    expect(catalog?.skillCount).toBe(1)

    patchUiState({ sid: 'elsewhere' })
    await loadSlashCatalog(rpc, 'elsewhere', setCatalog)
    expect(catalog?.canon[REPO_SKILL]).toBeUndefined()

    expect(requested).toEqual([undefined, 'repo', 'elsewhere'])
  })

  it('drops a reply that arrives after the live session has changed', async () => {
    let release = () => {}

    const gate = new Promise<void>(resolve => {
      release = resolve
    })

    const fake = async (_method: string, params: { session_id?: string } = {}) => {
      await gate

      return catalogFor(params.session_id)
    }

    // SAFETY: the fake answers only commands.catalog, the one method loadSlashCatalog calls, with that method's result type.
    const rpc = fake as GatewayRpc

    patchUiState({ sid: 'repo' })
    const pending = loadSlashCatalog(rpc, 'repo', setCatalog)

    patchUiState({ sid: 'elsewhere' })
    release()
    await pending

    expect(catalog).toBeNull()
  })
})
