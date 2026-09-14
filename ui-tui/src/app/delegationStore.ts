import type { DelegationStatusResult } from '@hermes/shared/gateway-events'
import { atom } from 'nanostores'

export interface DelegationState {
  // Last known caps from `delegation.status` RPC.  null until fetched.
  maxConcurrentChildren: null | number
  maxSpawnDepth: null | number
  // True when spawning is globally paused (see tools/delegate_tool.py).
  paused: boolean
  // Monotonic clock of the last successful status fetch.
  updatedAt: null | number
}

const buildState = (): DelegationState => ({
  maxConcurrentChildren: null,
  maxSpawnDepth: null,
  paused: false,
  updatedAt: null
})

export const $delegationState = atom<DelegationState>(buildState())

export const getDelegationState = () => $delegationState.get()

export const patchDelegationState = (next: Partial<DelegationState>) =>
  $delegationState.set({ ...$delegationState.get(), ...next })

export const resetDelegationState = () => $delegationState.set(buildState())

// ── Overlay accordion open-state ──────────────────────────────────────
//
// Lifted out of OverlaySection's local useState so collapse choices
// survive:
//   - navigating to a different subagent (Detail remounts)
//   - switching list ↔ detail mode (Detail unmounts in list mode)
//   - walking history (←/→)
// Keyed by section title; missing entries fall back to the section's
// `defaultOpen` prop.

export const $overlaySectionsOpen = atom<Record<string, boolean>>({})

export const toggleOverlaySection = (title: string, defaultOpen: boolean) => {
  const state = $overlaySectionsOpen.get()
  const current = title in state ? state[title]! : defaultOpen

  $overlaySectionsOpen.set({ ...state, [title]: !current })
}

export const getOverlaySectionOpen = (title: string, defaultOpen: boolean): boolean => {
  const state = $overlaySectionsOpen.get()

  return title in state ? state[title]! : defaultOpen
}

/** Merge a `delegation.status` result into the store. */
export const applyDelegationStatus = (r: DelegationStatusResult) =>
  patchDelegationState({
    maxConcurrentChildren: r.max_concurrent_children,
    maxSpawnDepth: r.max_spawn_depth,
    paused: r.paused,
    updatedAt: Date.now()
  })

/** `delegation.pause` answers the new flag alone; the caps are unchanged. */
export const applyDelegationPaused = (paused: boolean) => patchDelegationState({ paused, updatedAt: Date.now() })
