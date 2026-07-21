import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'
import { patchUiState } from './uiStore.js'

const buildOverlayState = (): OverlayState => ({
  agents: false,
  agentsInitialHistoryIndex: 0,
  approval: null,
  billing: null,
  clarify: null,
  confirm: null,
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
    journey: $overlayState.get().journey,
    modelPicker: $overlayState.get().modelPicker,
    petPicker: $overlayState.get().petPicker,
    pluginsHub: $overlayState.get().pluginsHub,
    sessions: $overlayState.get().sessions,
    skillsHub: $overlayState.get().skillsHub
  })

/** Prompt overlays whose answer is a backend round-trip (`*.respond` RPC). */
export type PromptOverlayKind = 'approval' | 'clarify' | 'secret' | 'sudo'

/**
 * Resolve an answered prompt overlay race-safely and report whether a *newer*
 * same-kind prompt is now active.
 *
 * Backend prompts are FIFO but their response RPCs are async. The backend can
 * emit prompt B (same kind, fresh requestId) right after removing prompt A yet
 * before A's `*.respond` round-trip resolves — so by the time A's late ACK
 * lands, B may already own the overlay slot. Clearing the slot unconditionally
 * (the old behaviour) erased B and left its backend waiter with no visible
 * prompt, and flipped the status back to `running…`, hiding that B still needs
 * input.
 *
 * So we clear the slot ONLY while the live overlay still carries the answered
 * requestId, and reset the status to `running…` only when no interactive
 * prompt of any kind remains. `supersededByNewer` reports same-kind ownership
 * for callers whose branch-specific behavior needs it (clarify cancellation,
 * for example, never resets status).
 * Mirrors the `sudo.expire` / `secret.expire` guards in createGatewayEventHandler.
 */
export function resolveAnsweredPrompt(
  kind: PromptOverlayKind,
  requestId: string,
  { resetStatus = true }: { resetStatus?: boolean } = {}
): { supersededByNewer: boolean } {
  patchOverlayState(prev =>
    (prev[kind] as null | { requestId: string })?.requestId === requestId ? { ...prev, [kind]: null } : prev
  )

  const state = getOverlayState()
  const live = state[kind] as null | { requestId: string }
  const supersededByNewer = live != null && live.requestId !== requestId
  const hasInteractivePrompt = Boolean(state.approval || state.clarify || state.secret || state.sudo)

  // A different prompt kind may also have arrived while this RPC was in
  // flight. Its waiting status owns the single status line just as strongly as
  // a same-kind successor, so never replace it with `running…`.
  if (resetStatus && !hasInteractivePrompt) {
    patchUiState({ status: 'running…' })
  }

  return { supersededByNewer }
}
