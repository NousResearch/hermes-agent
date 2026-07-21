import { describe, expect, it } from 'vitest'

import { rankPets } from '../components/petPicker.js'

describe('pet picker fuzzy filtering', () => {
  const pets = [
    { curated: true, displayName: 'Space Cat', installed: false, slug: 'space-cat' },
    { displayName: 'Rubber Duck', installed: true, slug: 'rubber-duck' },
    { displayName: 'Clawd Placeholder', installed: true, slug: 'clawd-test' }
  ]

  it('keeps active and installed pets ahead when no query is active', () => {
    expect(rankPets(pets, '', true, 'space-cat').map(pet => pet.slug)).toEqual(['space-cat', 'rubber-duck'])
  })

  it('supports fuzzy subsequence matches across display name and slug', () => {
    expect(rankPets(pets, 'rbdk', false, '').map(pet => pet.slug)).toEqual(['rubber-duck'])
    expect(rankPets(pets, 'sp cat', false, '').map(pet => pet.slug)).toEqual(['space-cat'])
  })

  it('continues to hide clawd placeholders', () => {
    expect(rankPets(pets, 'clawd', false, '')).toEqual([])
  })
})
