import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'

function buildOverlayState(): OverlayState {
  return {
    approval: null,
    clarify: null,
    modelPicker: false,
    pager: null,
    picker: false,
    secret: null,
    sudo: null
  }
}

export const $overlayState = atom<OverlayState>(buildOverlayState())

export const $isBlocked = computed($overlayState, state =>
  Boolean(
    state.approval || state.clarify || state.modelPicker || state.pager || state.picker || state.secret || state.sudo
  )
)

export function getOverlayState() {
  return $overlayState.get()
}

export function patchOverlayState(next: Partial<OverlayState> | ((state: OverlayState) => OverlayState)) {
  if (typeof next === 'function') {
    $overlayState.set(next($overlayState.get()))

    return
  }

  $overlayState.set({ ...$overlayState.get(), ...next })
}

export function resetOverlayState() {
  $overlayState.set(buildOverlayState())
}
