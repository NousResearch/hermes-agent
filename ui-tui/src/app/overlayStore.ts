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
  connection: null,
  connectors: false,
  ambient: [],
  widget: null,
  journey: false,
  modelPicker: false,
  pager: null,
  petPicker: false,
  pluginsHub: false,
  secret: null,
  vaultCode: null,
  vaultSaveLogin: null,
  vaultUnlock: null,
  sessions: false,
  skillsHub: false,
  subscription: null,
  sudo: null
})

export const $overlayState = atom<OverlayState>(buildOverlayState())

// Every overlay that suspends text input. `ambient` is the one non-blocking key.
const BLOCKING_KEYS = [
  'agents',
  'approval',
  'billing',
  'clarify',
  'confirm',
  'connection',
  'connectors',
  'journey',
  'modelPicker',
  'pager',
  'petPicker',
  'pluginsHub',
  'secret',
  'sessions',
  'skillsHub',
  'subscription',
  'sudo',
  'vaultCode',
  'vaultSaveLogin',
  'vaultUnlock',
  'widget'
] as const satisfies readonly (keyof OverlayState)[]

export const $isBlocked = computed($overlayState, overlay => BLOCKING_KEYS.some(key => Boolean(overlay[key])))

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
 *   `sessions`, `skillsHub`, `pluginsHub`, `connectors` — but ONLY when the rule sits at
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
    overlay.connectors ||
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

/**
 * SINGLE SOURCE for the credential prompts (sudo, secret, the vault cards):
 * Esc / Ctrl+C decline them, the tab title and pet read them as "waiting on
 * you". Add new credential prompts HERE so no gate silently misses one.
 */
export const SENSITIVE_PROMPTS = ['sudo', 'secret', 'vaultUnlock', 'vaultSaveLogin', 'vaultCode'] as const

export const hasSensitivePrompt = (overlay: Pick<OverlayState, (typeof SENSITIVE_PROMPTS)[number]>): boolean =>
  SENSITIVE_PROMPTS.some(key => Boolean(overlay[key]))

/** The agent's questions and confirms, which take the keyboard from any open panel. */
export const PROMPT_KEYS = ['approval', 'clarify', 'confirm', 'connection', ...SENSITIVE_PROMPTS] as const

export const hasPromptOpen = (overlay: OverlayState): boolean => PROMPT_KEYS.some(key => Boolean(overlay[key]))

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
 *
 * `connection` is preserved too: the card belongs to a backend operation that outlives the turn's
 * idle edge, and only the operation's own settlement may close it.
 */
export const resetFlowOverlays = () =>
  $overlayState.set({
    ...buildOverlayState(),
    agents: $overlayState.get().agents,
    agentsInitialHistoryIndex: $overlayState.get().agentsInitialHistoryIndex,
    ambient: $overlayState.get().ambient,
    connection: $overlayState.get().connection,
    connectors: $overlayState.get().connectors,
    widget: $overlayState.get().widget,
    journey: $overlayState.get().journey,
    modelPicker: $overlayState.get().modelPicker,
    petPicker: $overlayState.get().petPicker,
    pluginsHub: $overlayState.get().pluginsHub,
    sessions: $overlayState.get().sessions,
    skillsHub: $overlayState.get().skillsHub
  })
