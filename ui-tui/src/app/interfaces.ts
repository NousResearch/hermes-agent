import type { MouseTrackingMode, ScrollBoxHandle } from '@hermes/ink'
import type { MutableRefObject, ReactNode, RefObject, SetStateAction } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  BillingCardInfo,
  BillingMutationResponse,
  BillingStateResponse,
  ImageAttachResponse,
  SessionCloseResponse,
  SubscriptionPreviewResponse,
  SubscriptionStateResponse,
  SubscriptionUpgradeResponse
} from '../gatewayTypes.js'
import type { ParsedVoiceRecordKey } from '../lib/platform.js'
import type { RpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import type {
  ApprovalReq,
  ClarifyReq,
  ConfirmReq,
  DetailsMode,
  Msg,
  PanelSection,
  SecretReq,
  SectionVisibility,
  SessionInfo,
  SlashCatalog,
  SudoReq,
  Usage
} from '../types.js'

export interface StateSetter<T> {
  (value: SetStateAction<T>): void
}

export type StatusBarMode = 'bottom' | 'off' | 'top'

export type BusyInputMode = 'interrupt' | 'queue' | 'steer'

export type NoticeLevel = 'error' | 'info' | 'success' | 'warn'

// Credits/usage notice surfaced in the status bar. Shape is snake_case to
// match the gateway WS wire (`notification.show` payload) and the existing
// `Usage` type — no camelCase mapping layer. The `text` already carries its
// own leading glyph (⚠ • ✕ ✓) from the Python policy, so the renderer only
// colours it by `level` and never adds another glyph.
export interface Notice {
  id?: string
  key?: string
  kind?: 'sticky' | 'ttl'
  level?: NoticeLevel
  text: string
  ttl_ms?: null | number
}

// Single source of truth for indicator style names.  Union type is
// derived from this tuple so adding/removing a style only touches one
// line — `useConfigSync` (validation) and `session.ts` (slash arg
// validation + usage hint) both import it.
export const INDICATOR_STYLES = ['ascii', 'emoji', 'kaomoji', 'unicode'] as const
export type IndicatorStyle = (typeof INDICATOR_STYLES)[number]
export const DEFAULT_INDICATOR_STYLE: IndicatorStyle = 'kaomoji'

export interface SelectionApi {
  captureScrolledRows: (firstRow: number, lastRow: number, side: 'above' | 'below') => void
  clearSelection: () => void
  copySelection: () => Promise<string>
  copySelectionNoClear: () => Promise<string>
  getState: () => unknown
  version: () => number
  shiftAnchor: (dRow: number, minRow: number, maxRow: number) => void
  shiftSelection: (dRow: number, minRow: number, maxRow: number) => void
}

export interface CompletionItem {
  display: string
  meta?: string
  text: string
}

export interface GatewayRpc {
  <T extends RpcResult = RpcResult>(method: string, params?: Record<string, unknown>): Promise<null | T>
}

export interface GatewayServices {
  gw: GatewayClient
  rpc: GatewayRpc
}

export interface GatewayProviderProps {
  children: ReactNode
  value: GatewayServices
}

// ── Billing overlay (Phase 2b: full-modal TUI parity) ────────────────
// The /billing command no longer parses sub-commands; bare `/billing`
// fetches `billing.state` and opens this overlay.  The overlay is a small
// state machine (overview → buy|autoreload|limit → confirm) that performs
// the SAME RPCs as the old slash flows (billing.charge / charge_status /
// auto_reload / step_up).  Backend is unchanged & shared with the CLI.

export type BillingScreen = 'autoreload' | 'buy' | 'confirm' | 'limit' | 'overview' | 'stepup'

/** Outcome of a charge attempt — lets the overlay route without tearing down. */
export type BillingChargeOutcome =
  | 'submitted' // 202 accepted; settlement is reported via transcript lines
  | 'needs_remote_spending' // insufficient_scope → route to the stepup screen
  | 'error' // any other failure (already surfaced via sys)

/**
 * The functions the overlay needs to talk to the gateway and emit
 * transcript lines.  Built once in `billing.ts` (closing over the live
 * SlashRunCtx) and stashed in the overlay slot, mirroring how a ConfirmReq
 * stashes its `onConfirm` closure.  Keeps all RPC + error-mapping logic in
 * billing.ts (single source of truth) — the overlay only renders + routes.
 */
export interface BillingOverlayCtx {
  /** Run `billing.auto_reload` (enabled/threshold/top_up) → resolve ok/false. */
  applyAutoReload: (enabled: boolean, threshold?: number, topUp?: number) => Promise<boolean>
  /**
   * Submit `billing.charge` for `amount` and poll to settlement. Resolves a
   * discriminated outcome so the overlay can route to the resumable step-up on
   * `needs_remote_spending` instead of tearing down. Settlement/most errors are
   * still reported via transcript lines (the poll is non-blocking).
   */
  charge: (amount: string, idempotencyKey?: string) => Promise<BillingChargeOutcome>
  /**
   * Run the `billing.step_up` device flow (allow Remote Spending). Resolves
   * `true` when the grant lands. The browser opens via the gateway's
   * out-of-band `billing.step_up.verification` event — the overlay just awaits.
   */
  requestRemoteSpending: () => Promise<boolean>
  /** Open the portal in the browser + echo a transcript line. */
  openPortal: (url: string) => void
  /**
   * Re-fetch billing state (`billing.state`) — used by the add-card path's
   * "I've added it — check again" so a card saved on the portal appears without
   * re-running /topup. Resolves null on failure (caller keeps the old state).
   */
  refreshState: () => Promise<BillingStateResponse | null>
  /** Emit a transcript system line. */
  sys: (text: string) => void
  /** Validate a custom amount against state bounds + 2dp (mirrors the server). */
  validate: (raw: string) => { amount?: string; error?: string }
}

/** Pending confirm built when leaving the buy/autoreload screen. */
export interface BillingPendingCharge {
  amount: string
  /**
   * Stable idempotency key for THIS purchase, minted when the amount is chosen.
   * Reused across the step-up replay so a re-charge after the grant dedups
   * server-side (and a double-submit collapses to one charge).
   */
  idempotencyKey?: string
}

export interface BillingOverlayState {
  ctx: BillingOverlayCtx
  /** Set when on the 'confirm' screen for a buy. */
  pendingCharge?: BillingPendingCharge | null
  screen: BillingScreen
  state: BillingStateResponse
}

// ── Subscription overlay (in-terminal plan change, V3) ──

// A small state machine: overview → picker → confirm → result, with a stepup
// screen spliced in on demand.
//   overview — plan + status, entry to the picker / resume / manage-on-portal.
//   picker   — the tier catalog (up/down direction hints; current tier shown,
//              not selectable).
//   confirm  — the previewed effect of the chosen change (charge $X now /
//              scheduled at date / no-op / blocked) + the apply action.
//   result   — the outcome, including an SCA/decline upgrade handed off to the
//              portal.
//   stepup   — reached when a mutation returns insufficient_scope: allows remote
//              spending in place, then auto-replays the held action.
export type SubscriptionScreen = 'confirm' | 'overview' | 'picker' | 'result' | 'stepup'

// The action held while the stepup screen allows remote spending, replayed after
// approval: re-preview a tier, re-apply the confirmed pending change, or re-resume.
export type SubscriptionStepUpRetry = { kind: 'apply' } | { kind: 'preview'; tierId: string } | { kind: 'resume' }

/** Outcome of a remote-spending step-up: granted, plus the typed denial (for copy). */
export interface StepUpResult {
  granted: boolean
  error?: string
  message?: string
}

export interface SubscriptionOverlayCtx {
  /**
   * Best-effort card lookup (`billing.state`) for the upgrade confirm — shows
   * WHICH card the upgrade will charge. Resolves null on any failure or when
   * the server doesn't say (older NAS): the confirm keeps its generic line.
   */
  fetchCard: () => Promise<BillingCardInfo | null>
  /** Build {portal}/manage-subscription?org_id=… locally and open it. Resolves ok/false. */
  openManageLink: () => Promise<boolean>
  /** Open an arbitrary portal recovery URL (e.g. an upgrade's SCA handoff). */
  openPortal: (url: string) => void
  /** Re-fetch subscription.state. */
  refreshState: () => Promise<SubscriptionStateResponse | null>
  /** POST /preview a change to `tierId` → the chargeless effect quote (or typed error). */
  preview: (tierId: string) => Promise<SubscriptionPreviewResponse | null>
  /** PUT pending-change: schedule a downgrade / same-price change to `tierId`. */
  scheduleChange: (tierId: string) => Promise<BillingMutationResponse | null>
  /** PUT pending-change: schedule a cancellation at period end. */
  scheduleCancellation: () => Promise<BillingMutationResponse | null>
  /** DELETE pending-change: clear a scheduled downgrade / cancellation (resume). */
  resume: () => Promise<BillingMutationResponse | null>
  /** POST /upgrade: charge the card on the subscription + flip the plan now. */
  upgrade: (tierId: string, idempotencyKey?: string) => Promise<SubscriptionUpgradeResponse | null>
  /**
   * Run the `billing.step_up` device flow (allow remote spending / "Remote
   * Spending"). Resolves `{granted}` plus the typed denial (`error`/`message`) so
   * the stepup screen shows the right recovery. The browser opens via the
   * gateway's out-of-band verification event — the stepup screen just awaits.
   */
  requestRemoteSpending: () => Promise<StepUpResult>
  /** Emit a transcript system line. */
  sys: (text: string) => void
}

/** What the confirm screen is about to apply, plus its preview quote. */
export interface SubscriptionPendingChange {
  /** The target tier (null for a cancellation). */
  targetTierId: string | null
  /** How it will be applied — drives which ctx call confirm makes. */
  kind: 'cancellation' | 'tier_change' | 'upgrade'
  /** The preview quote shown on confirm (null = the quote call failed). */
  preview?: null | SubscriptionPreviewResponse
  /**
   * Stable idempotency key for an upgrade charge, minted when confirm opens.
   * Reused on retry so a re-submit dedups server-side.
   */
  idempotencyKey?: string
}

/** The outcome rendered on the result screen. */
export interface SubscriptionResult {
  message: string
  ok: boolean
  /** Set on a successful upgrade; drives the ResultScreen apply-poll. */
  pendingTierId?: null | string
  /** A portal URL to finish an SCA/declined upgrade, when present. */
  recoveryUrl?: null | string
}

export interface SubscriptionOverlayState {
  ctx: SubscriptionOverlayCtx
  /** Set on the 'confirm' screen: the change being confirmed + its preview. */
  pending?: null | SubscriptionPendingChange
  /** Set on the 'result' screen: the outcome to render. */
  result?: null | SubscriptionResult
  screen: SubscriptionScreen
  state: SubscriptionStateResponse
  /** Held while on the 'stepup' screen: the action to replay once the grant lands. */
  stepUpRetry?: null | SubscriptionStepUpRetry
}

export interface OverlayState {
  agents: boolean
  agentsInitialHistoryIndex: number
  approval: ApprovalReq | null
  billing: BillingOverlayState | null
  clarify: ClarifyReq | null
  confirm: ConfirmReq | null
  journey: boolean
  modelPicker: boolean | { refresh?: boolean }
  pager: null | PagerState
  petPicker: boolean
  pluginsHub: boolean
  secret: null | SecretReq
  sessions: boolean
  skillsHub: boolean
  subscription: SubscriptionOverlayState | null
  sudo: null | SudoReq
}

export interface PagerState {
  lines: string[]
  offset: number
  title?: string
}

export interface TranscriptRow {
  index: number
  key: string
  msg: Msg
}

export interface UiState {
  bgTasks: Set<string>
  busy: boolean
  busyInputMode: BusyInputMode
  compact: boolean
  detailsMode: DetailsMode
  detailsModeCommandOverride: boolean
  info: null | SessionInfo
  liveSessionCount: number
  inlineDiffs: boolean
  mouseTracking: MouseTrackingMode
  notice: Notice | null
  pasteCollapseLines: number
  pasteCollapseChars: number

  sections: SectionVisibility
  sessionTitle: string
  showReasoning: boolean
  indicatorStyle: IndicatorStyle
  sid: null | string
  status: string
  statusBar: StatusBarMode
  streaming: boolean
  theme: Theme
  usage: Usage
}

export interface VirtualHistoryState {
  bottomSpacer: number
  end: number
  measureRef: (key: string) => (el: unknown) => void
  offsets: ArrayLike<number>
  start: number
  topSpacer: number
}

export interface ComposerPasteResult {
  cursor: number
  value: string
}

export type MaybePromise<T> = Promise<T> | T

export interface ComposerActions {
  clearIn: () => void
  dequeue: () => string | undefined
  enqueue: (text: string) => void
  handleTextPaste: (event: PasteEvent) => MaybePromise<ComposerPasteResult | null>
  openEditor: () => Promise<void>
  pushHistory: (text: string) => void
  removeQueue: (index: number) => void
  replaceQueue: (index: number, text: string) => void
  setCompIdx: StateSetter<number>
  setHistoryIdx: StateSetter<null | number>
  setInput: StateSetter<string>
  setInputBuf: StateSetter<string[]>
  setPasteSnips: StateSetter<PasteSnippet[]>
  setQueueEdit: (index: null | number) => void
  syncQueue: () => void
}

export interface ComposerRefs {
  historyDraftRef: MutableRefObject<string>
  historyRef: MutableRefObject<string[]>
  queueEditRef: MutableRefObject<null | number>
  queueRef: MutableRefObject<string[]>
  submitRef: MutableRefObject<(value: string) => void>
}

export interface ComposerState {
  compIdx: number
  compReplace: number
  completions: CompletionItem[]
  historyIdx: null | number
  input: string
  inputBuf: string[]
  pasteSnips: PasteSnippet[]
  queueEditIdx: null | number
  queuedDisplay: string[]
}

export interface UseComposerStateOptions {
  gw: GatewayClient
  onClipboardPaste: (quiet?: boolean) => Promise<void> | void
  onImageAttached?: (info: ImageAttachResponse) => void
  submitRef: MutableRefObject<(value: string) => void>
}

export interface UseComposerStateResult {
  actions: ComposerActions
  refs: ComposerRefs
  state: ComposerState
}

export interface InputHandlerActions {
  answerClarify: (answer: string) => void
  appendMessage: (msg: Msg) => void
  die: () => void
  dispatchSubmission: (full: string) => void
  guardBusySessionSwitch: (what?: string) => boolean
  newSession: (msg?: string, title?: string) => void
  sys: (text: string) => void
}

export interface InputHandlerContext {
  actions: InputHandlerActions
  composer: {
    actions: ComposerActions
    refs: ComposerRefs
    state: ComposerState
  }
  gateway: GatewayServices
  terminal: {
    hasSelection: boolean
    scrollRef: RefObject<null | ScrollBoxHandle>
    scrollWithSelection: (delta: number) => void
    selection: SelectionApi
    stdout?: NodeJS.WriteStream
  }
  voice: {
    enabled: boolean
    recordKey: ParsedVoiceRecordKey
    recording: boolean
    setProcessing: StateSetter<boolean>
    setRecording: StateSetter<boolean>
    setVoiceEnabled: StateSetter<boolean>
    setVoiceTts: StateSetter<boolean>
  }
  wheelStep: number
}

export interface InputHandlerResult {
  pagerPageSize: number
}

export interface GatewayEventHandlerContext {
  composer: {
    setInput: StateSetter<string>
  }
  gateway: GatewayServices
  session: {
    STARTUP_RESUME_ID: string
    colsRef: MutableRefObject<number>
    newSession: (msg?: string, title?: string) => void
    // Set by useMainApp's exit handler to the session that was live when the
    // gateway died unexpectedly; consumed once by the next `gateway.ready` so a
    // respawn resumes that session instead of forging a fresh one.
    recoverSidRef?: MutableRefObject<null | string>
    resetSession: () => void
    resumeById: (id: string) => void
    setCatalog: StateSetter<null | SlashCatalog>
  }
  submission: {
    submitRef: MutableRefObject<(value: string) => void>
  }
  system: {
    bellOnComplete: boolean
    stdout?: NodeJS.WriteStream
    sys: (text: string) => void
  }
  transcript: {
    appendMessage: (msg: Msg) => void
    panel: (title: string, sections: PanelSection[]) => void
    setHistoryItems: StateSetter<Msg[]>
  }
  voice: {
    setProcessing: StateSetter<boolean>
    setRecording: StateSetter<boolean>
    setVoiceEnabled: StateSetter<boolean>
    setVoiceTts: StateSetter<boolean>
  }
}

export interface SlashHandlerContext {
  composer: {
    enqueue: (text: string) => void
    hasSelection: boolean
    openEditor: () => Promise<void>
    paste: (quiet?: boolean) => void
    queueRef: MutableRefObject<string[]>
    selection: SelectionApi
    setInput: StateSetter<string>
  }
  gateway: GatewayServices
  local: {
    catalog: null | SlashCatalog
    getHistoryItems: () => Msg[]
    getLastUserMsg: () => string
    maybeWarn: (value: unknown) => void
    setCatalog: StateSetter<null | SlashCatalog>
  }
  session: {
    closeSession: (targetSid?: null | string) => Promise<unknown>
    die: () => void
    dieWithCode: (code: number) => void
    guardBusySessionSwitch: (what?: string) => boolean
    newLiveSession: (msg?: string, title?: string) => void
    newSession: (msg?: string, title?: string) => void
    resetVisibleHistory: (info?: null | SessionInfo) => void
    resumeById: (id: string) => void
    setSessionStartedAt: StateSetter<number>
  }
  slashFlightRef: MutableRefObject<number>
  transcript: {
    page: (text: string, title?: string) => void
    panel: (title: string, sections: PanelSection[]) => void
    send: (text: string) => void
    setHistoryItems: StateSetter<Msg[]>
    sys: (text: string) => void
    trimLastExchange: (items: Msg[]) => Msg[]
  }
  voice: {
    setVoiceEnabled: StateSetter<boolean>
    setVoiceRecordKey: (v: ParsedVoiceRecordKey) => void
    setVoiceTts: StateSetter<boolean>
  }
}

export interface AppLayoutActions {
  answerApproval: (choice: string) => void
  answerClarify: (answer: string) => void
  answerSecret: (value: string) => void
  answerSudo: (pw: string) => void
  clearSelection: () => void
  activateLiveSession: (id: string) => void
  closeLiveSession: (id: string) => Promise<null | SessionCloseResponse>
  newLiveSession: () => void
  newPromptSession: (prompt: string, modelArg?: string) => void
  onModelSelect: (value: string) => void
  resumeById: (id: string) => void
  setStickyPrompt: (value: string) => void
}

export interface AppLayoutComposerProps {
  cols: number
  compIdx: number
  completions: CompletionItem[]
  empty: boolean
  handleTextPaste: (event: PasteEvent) => MaybePromise<ComposerPasteResult | null>
  input: string
  inputBuf: string[]
  pagerPageSize: number
  queueEditIdx: null | number
  queuedDisplay: string[]
  submit: (value: string) => void
  updateInput: StateSetter<string>
  voiceRecordKey: ParsedVoiceRecordKey
}

export interface AppLayoutProgressProps {
  showProgressArea: boolean
}

export interface AppLayoutStatusProps {
  cwdLabel: string
  goodVibesTick: number
  lastTurnEndedAt: null | number
  sessionStartedAt: null | number
  showStickyPrompt: boolean
  statusColor: string
  stickyPrompt: string
  turnStartedAt: null | number
  voiceLabel: string
}

export interface AppLayoutTranscriptProps {
  historyItems: Msg[]
  scrollRef: RefObject<null | ScrollBoxHandle>
  virtualHistory: VirtualHistoryState
  virtualRows: TranscriptRow[]
}

export interface AppLayoutProps {
  actions: AppLayoutActions
  composer: AppLayoutComposerProps
  mouseTracking: MouseTrackingMode
  progress: AppLayoutProgressProps
  status: AppLayoutStatusProps
  transcript: AppLayoutTranscriptProps
}

export interface AppOverlaysProps {
  cols: number
  compIdx: number
  completions: CompletionItem[]
  onApprovalChoice: (choice: string) => void
  onClarifyAnswer: (value: string) => void
  onActiveSessionSelect: (sessionId: string) => void
  onActiveSessionClose: (sessionId: string) => Promise<null | SessionCloseResponse>
  onModelSelect: (value: string) => void
  onNewLiveSession: () => void
  onNewPromptSession: (prompt: string, modelArg?: string) => void
  onResumeSelect: (sessionId: string) => void
  onSecretSubmit: (value: string) => void
  onSudoSubmit: (pw: string) => void
  pagerPageSize: number
}

export interface PasteSnippet {
  label: string
  path?: string
  text: string
}
