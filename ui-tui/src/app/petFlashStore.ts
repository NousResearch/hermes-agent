import { atom } from 'nanostores'

import type { PetState } from './usePet.js'

interface PetFlash {
  state: PetState
  until: number
}

// Transient reaction beats (wave/jump/failed) the pet shows for a moment at
// turn end before falling back to its steady state. The gateway event handler
// sets these; usePet reads them with priority over the derived state.
export const $petFlash = atom<PetFlash | null>(null)

export const flashPet = (state: PetState, ms = 1600) => $petFlash.set({ state, until: Date.now() + ms })

// The floating pet's footprint, or null when no pet is shown. The transcript
// keeps its text clear of the pet responsively: on wide terminals it reserves a
// right gutter (`width`) so lines wrap to the pet's LEFT; on narrow terminals it
// reserves bottom rows (`height`) so lines stay full-width and sit ABOVE it.
export const $petBox = atom<{ width: number; height: number } | null>(null)
