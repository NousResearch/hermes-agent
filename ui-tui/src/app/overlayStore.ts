import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'

const buildOverlayState = (): OverlayState => ({
  agents: false,
  agentsInitialHistoryIndex: 0,
  approval: null,
  billing: null,
  clarify: null,
  confirm: null,
  dialog: null,
  gridTest: null,
  journey: false,
  modelPicker: false,
  pager: null,
  petPicker: false,
  pluginsHub: false,
  secret: null,
  sessions: false,
  skillsHub: false,
  subscription: null,
  sudo: null
})

export const $overlayState = atom<OverlayState>(buildOverlayState())

export const $isBlocked = computed(
  $overlayState,
  ({
    agents,
    approval,
    billing,
    clarify,
    confirm,
    dialog,
    gridTest,
    journey,
    modelPicker,
    pager,
    petPicker,
    pluginsHub,
    secret,
    sessions,
    skillsHub,
    subscription,
    sudo
  }) =>
    Boolean(
      agents ||
      approval ||
      billing ||
      clarify ||
      confirm ||
      dialog ||
      gridTest ||
      journey ||
      modelPicker ||
      pager ||
      petPicker ||
      pluginsHub ||
      secret ||
      sessions ||
      skillsHub ||
      subscription ||
      sudo
    )
)

export const getOverlayState = () => $overlayState.get()

export const patchOverlayState = (next: Partial<OverlayState> | ((state: OverlayState) => OverlayState)) =>
  $overlayState.set(typeof next === 'function' ? next($overlayState.get()) : { ...$overlayState.get(), ...next })

/** Full reset — used by session/turn teardown and tests. */
export const resetOverlayState = () => $overlayState.set(buildOverlayState())

/**
 * Soft reset: drop FLOW-scoped overlays (approval / clarify / confirm / sudo
 * / secret / pager) but PRESERVE user-toggled ones — agents dashboard, model
 * picker, skills hub, sessions overlay.  Those are opened deliberately and
 * shouldn't vanish when a turn ends.  Called from turnController.idle() on
 * every turn completion / interrupt; the old "reset everything" behaviour
 * silently closed /agents the moment delegation finished.
 */
export const resetFlowOverlays = () =>
  $overlayState.set({
    ...buildOverlayState(),
    agents: $overlayState.get().agents,
    agentsInitialHistoryIndex: $overlayState.get().agentsInitialHistoryIndex,
    dialog: $overlayState.get().dialog,
    gridTest: $overlayState.get().gridTest,
    journey: $overlayState.get().journey,
    modelPicker: $overlayState.get().modelPicker,
    petPicker: $overlayState.get().petPicker,
    pluginsHub: $overlayState.get().pluginsHub,
    sessions: $overlayState.get().sessions,
    skillsHub: $overlayState.get().skillsHub
  })
