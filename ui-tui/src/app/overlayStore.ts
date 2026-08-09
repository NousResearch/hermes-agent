import { atom, computed } from 'nanostores'

import type { OverlayState } from './interfaces.js'
import { $uiState } from './uiStore.js'

const buildOverlayState = (): OverlayState => ({
  agents: false,
  agentsInitialHistoryIndex: 0,
  approval: null,
  billing: null,
  clarify: null,
  confirm: null,
  ambient: [],
  widget: null,
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
    sudo,
    widget
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
      sudo ||
      widget
    )
)

/**
 * Does an open overlay actually PAINT OVER the status rule?
 *
 * Deliberately NOT `$isBlocked`.  That aggregate answers a different
 * question — "is text input suspended" — and `appLayout` uses it only to hide
 * the input rows (`appLayout.tsx:384`).  `StatusRulePane` is rendered OUTSIDE
 * that guard (`appLayout.tsx:365` for `at="top"`, `:449` for `at="bottom"`),
 * so most `$isBlocked` fields leave the rule fully visible.
 *
 * Occluding (included here):
 *
 * - `widget` — the modal widget slot renders at viewport level
 *   (`ActiveWidgetSlot`, `sdk/host.tsx:209`, outside the ComposerPane
 *   subtree) so it can anchor the full-screen absolute `Overlay`
 *   (`components/overlay.tsx`) against the whole terminal.
 * - The FloatingOverlays set — `modelPicker`, `pager`, `petPicker`,
 *   `sessions`, `skillsHub`, `pluginsHub` — but ONLY when the rule sits at
 *   the top.  That panel is `position="absolute" bottom="100%"` inside
 *   ComposerPane's relative Box (`appOverlays.tsx:387`), so it grows UPWARD
 *   over the `at="top"` rule and never reaches the `at="bottom"` one.
 *
 * NOT occluding (deliberately excluded):
 *
 * - The PromptZone flow states — `approval`, `billing`, `subscription`,
 *   `confirm`, `clarify`, `sudo`, `secret` (`appOverlays.tsx:58-162`).  They
 *   render in NORMAL FLOW above ComposerPane (`appLayout.tsx:553-568`): they
 *   push content down, they do not cover it.  The rule stays on screen and
 *   its clock must keep running.
 * - `agents` and `journey`.  They unmount the entire ComposerPane subtree
 *   (`appLayout.tsx:553`), so `StatusRule` unmounts with it and React's own
 *   effect cleanup clears the intervals.  Gating on them would be dead code.
 * - `ambient` — a glanceable in-flow dock that reserves its own rows.
 * - Composer completions.  They share the FloatingOverlays grid and do
 *   occlude, but they are a render prop rather than store state and they
 *   change on every keystroke; re-arming a 1s interval per character would
 *   restart the countdown each time and starve the tick outright — strictly
 *   worse than the churn this gate removes.
 *
 * `statusBar: 'off'` needs no branch: `StatusRulePane` returns null for both
 * slots, so the timers are never mounted in the first place.
 */
/**
 * True when any floating overlay PANEL is open (widget overlays and inline
 * completions are separate concerns — see $isStatusRuleOccluded for why
 * completions never occlude the status rule).
 *
 * SINGLE SOURCE for the floating-panel kind set: consumed by both
 * FloatingOverlays' render gate (plus completions, which it adds locally)
 * and $isStatusRuleOccluded's top-statusbar occlusion arm. Add new floating
 * panels HERE so the timer gate can't silently miss them.
 */
export const hasFloatingPanel = (overlay: OverlayState): boolean =>
  Boolean(
    overlay.modelPicker ||
    overlay.pager ||
    overlay.petPicker ||
    overlay.pluginsHub ||
    overlay.sessions ||
    overlay.skillsHub
  )

export const $isStatusRuleOccluded = computed([$overlayState, $uiState], (overlay, ui) =>
  Boolean(overlay.widget || (ui.statusBar === 'top' && hasFloatingPanel(overlay)))
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
    ambient: $overlayState.get().ambient,
    widget: $overlayState.get().widget,
    journey: $overlayState.get().journey,
    modelPicker: $overlayState.get().modelPicker,
    petPicker: $overlayState.get().petPicker,
    pluginsHub: $overlayState.get().pluginsHub,
    sessions: $overlayState.get().sessions,
    skillsHub: $overlayState.get().skillsHub
  })
