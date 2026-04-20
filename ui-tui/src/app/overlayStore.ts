import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'

const buildOverlayState = (): OverlayState => ({
  approval: null,
  clarify: null,
  modelPicker: false,
  pager: null,
  picker: false,
  secret: null,
  setupWizard: false,
  sudo: null,
  swarm: false
})

export const $overlayState = atom<OverlayState>(buildOverlayState())

export const $isBlocked = computed($overlayState, ({ approval, clarify, modelPicker, pager, picker, secret, setupWizard, sudo }) =>
  Boolean(approval || clarify || modelPicker || pager || picker || secret || setupWizard || sudo)
)

export const getOverlayState = () => $overlayState.get()

export const patchOverlayState = (next: Partial<OverlayState> | ((state: OverlayState) => OverlayState)) =>
  $overlayState.set(typeof next === 'function' ? next($overlayState.get()) : { ...$overlayState.get(), ...next })

export const resetOverlayState = () => $overlayState.set(buildOverlayState())
