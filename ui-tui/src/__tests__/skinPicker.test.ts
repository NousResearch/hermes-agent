import { describe, expect, it } from 'vitest'

import { filterSkinOptions, skinAgentLabel } from '../components/skinPicker.js'

const skins = [
  { description: 'Ocean-god theme', name: 'poseidon', source: 'builtin' },
  { description: 'Volcanic theme', name: 'charizard', source: 'builtin' },
  { description: 'Custom blue theme', name: 'blueprint', source: 'user' }
]

describe('skin picker helpers', () => {
  it('keeps every skin when the filter is empty', () => {
    expect(filterSkinOptions(skins, '')).toEqual(skins)
  })

  it('filters by skin name and description', () => {
    expect(filterSkinOptions(skins, 'ocean').map(skin => skin.name)).toEqual(['poseidon'])
    expect(filterSkinOptions(skins, 'blue').map(skin => skin.name)).toEqual(['blueprint'])
  })

  it('shows optional agent branding without pretending every skin renames Hermes', () => {
    expect(skinAgentLabel({ agent_name: 'Charizard Agent' })).toBe('agent: Charizard Agent')
    expect(skinAgentLabel({})).toBe('agent: Hermes Agent')
  })
})
