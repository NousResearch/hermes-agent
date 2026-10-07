import type { MouseTrackingMode, ScrollBoxHandle } from '@rabbit/ink'
import type { Usage } from '@rabbit/shared/gateway-events'
import type { MutableRefObject, ReactNode, RefObject, SetStateAction } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { SessionCloseResponse } from '../gatewayTypes.js'
import type { QueueItem } from '../hooks/useQueue.js'
import type { ParsedVoiceRecordKey } from '../lib/platform.js'
import type { RpcResult } from '../lib/rpc.js'
import type { ActiveWidget } from '../sdk/types.js'
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
  VaultCodeReq,
  VaultSaveLoginReq,
  VaultUnlockReq
} from '../types.js'

export interface StateSetter<T> {
  (value: SetStateAction<T>): void
}

export type StatusBarMode = 'bottom' | 'off' | 'top'

export type BatteryCategory = 'bad' | 'critical' | 'dim' | 'good' | 'warn'

// A single battery reading pushed from the Python gateway (`system.battery`).
// `available` is false on machines without a battery; `percent` is 0-100.
export interface BatteryInfo {
  available: boolean
  category: BatteryCategory
  percent: null | number
  plugged: null | boolean
}

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
  /** Completion class from the gateway; `skill` is the only kind offered for
   *  an inline `/skill` reference typed mid-message. */
  kind?: string
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

export interface ConnectionOverlayState {
  opId: string
}

export interface OverlayState {
  agents: boolean
  agentsInitialHistoryIndex: number
  approval: ApprovalReq | null
  clarify: ClarifyReq | null
  confirm: ConfirmReq | null
  connection: ConnectionOverlayState | null
  /** Ambient widget apps — glanceable dock, non-blocking (never in $isBlocked). */
  ambient: ActiveWidget[]
  /** Modal widget app — owns input, blocks the composer. */
  widget: ActiveWidget | null
  journey: boolean
  modelPicker: boolean | { refresh?: boolean }
  pager: null | PagerState
  petPicker: boolean
  pluginsHub: boolean
  secret: null | SecretReq
  vaultCode: null | VaultCodeReq
  vaultSaveLogin: null | VaultSaveLoginReq
  vaultUnlock: null | VaultUnlockReq
  sessions: boolean
  skillsHub: boolean
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
  battery: boolean
  batteryStatus: BatteryInfo | null
  bgTasks: Set<string>
  busy: boolean
  busyInputMode: BusyInputMode
  compact: boolean
  // Context compaction in progress (idle/preflight/auto). Distinct from
  // `compact`, which is the /compact layout-density flag.
  compacting: boolean
  destructiveSlashConfirm: boolean
  detailsMode: DetailsMode
  detailsModeCommandOverride: boolean
  // Focus view (/focus) — display-only reduced-output mode. Drives the
  // persistent `◉ focus` status-bar badge; never affects request payloads.
  focusView: boolean
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
  // Durable session id (state.db row) of the live session — what session.resume
  // and the exit epilogue take. Kept apart from `info`, which producers replace
  // wholesale with payloads that may omit `stored_session_id`.
  storedSid: null | string
  // display.status_bar.fields — visibility filter for status-rule segments,
  // shared with the classic CLI bar. null = user has not customized (show
  // the default set).
  statusBarFields: null | ReadonlySet<string>
  streaming: boolean
  theme: Theme
  // `display.timestamps` — dim [HH:MM] labels on user/assistant transcript
  // rows, the same config key the classic CLI honors (#41531).
  timestamps: boolean
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
  /** Pull an image off the system clipboard in as a token. */
  attachClipboardImage: () => void
  /** Attach an image by path in as a token. */
  attachImagePath: (path: string) => void
  clearIn: () => void
  dequeue: () => string | undefined
  enqueue: (text: string, display?: string) => void
  handleTextPaste: (event: PasteEvent) => MaybePromise<ComposerPasteResult | null>
  openEditor: () => Promise<void>
  prependQueue: (item: QueueItem) => void
  pushHistory: (text: string) => void
  removeQueue: (index: number) => void
  setCompIdx: StateSetter<number>
  setComposerTokens: StateSetter<ComposerToken[]>
  setHistoryIdx: StateSetter<null | number>
  setInput: StateSetter<string>
  setInputBuf: StateSetter<string[]>
  setQueueEdit: (index: null | number) => void
  takeQueue: (index: number, editedDisplay?: string) => QueueItem | undefined
  /** Reconcile attached payloads against tokens still present in the text. */
  syncTokens: (value: string) => void
}

export interface ComposerRefs {
  historyDraftRef: MutableRefObject<string>
  historyRef: MutableRefObject<string[]>
  queueEditRef: MutableRefObject<null | number>
  queueRef: MutableRefObject<QueueItem[]>
  submitRef: MutableRefObject<(value: string) => void>
  tokensRef: MutableRefObject<ComposerToken[]>
}

export interface ComposerState {
  compIdx: number
  compReplace: number
  completions: CompletionItem[]
  historyIdx: null | number
  input: string
  inputBuf: string[]
  queueEditIdx: null | number
  queuedDisplay: string[]
  tokens: ComposerToken[]
}

export interface UseComposerStateOptions {
  gw: GatewayClient
  submitRef: MutableRefObject<(value: string) => void>
  sys: (text: string) => void
}

export interface UseComposerStateResult {
  actions: ComposerActions
  refs: ComposerRefs
  state: ComposerState
}

export interface InputHandlerActions {
  appendMessage: (msg: Msg) => void
  cancelClarify: () => void
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
    // Session carried across a transport loss or child exit, cleared after resume.
    recoverSidRef?: MutableRefObject<null | string>
    resetSession: () => void
    resumeById: (id: string) => Promise<void>
    setCatalog: StateSetter<null | SlashCatalog>
  }
  submission: {
    /** Submit text literally as a prompt — no slash/!/interpolation dispatch.
     *  Used for `-q` startup queries, which are arbitrary launcher-provided
     *  text (parity with one-shot's literal prompt handling). */
    submitLiteralRef: MutableRefObject<(value: string) => void>
    submitRef: MutableRefObject<(value: string) => void>
  }
  system: {
    bellOnComplete: boolean
    bellOnPrompt?: boolean
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
    attachClipboardImage: () => void
    attachImagePath: (path: string) => void
    enqueue: (text: string, display?: string) => void
    hasSelection: boolean
    openEditor: () => Promise<void>
    queueRef: MutableRefObject<QueueItem[]>
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
    send: (text: string, showUserMessage?: boolean, displayText?: string) => void
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
  answerClarifyQuestion: (qid: string, answer: string) => void
  answerSecret: (value: string) => void
  answerSudo: (pw: string) => void
  answerVaultCode: (code: string) => void
  answerVaultSaveLogin: (identifier: string, password: string) => void
  answerVaultUnlock: (password: string) => void
  cancelClarify: () => void
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
  sessionTitle: string
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
  onClarifyCancel: () => void
  onClarifyQuestionAnswer: (qid: string, value: string) => void
  onActiveSessionSelect: (sessionId: string) => void
  onActiveSessionClose: (sessionId: string) => Promise<null | SessionCloseResponse>
  onModelSelect: (value: string) => void
  onNewLiveSession: () => void
  onNewPromptSession: (prompt: string, modelArg?: string) => void
  onResumeSelect: (sessionId: string) => void
  onSecretSubmit: (value: string) => void
  onSudoSubmit: (pw: string) => void
  onVaultCodeSubmit: (code: string) => void
  onVaultSaveLoginSubmit: (identifier: string, password: string) => void
  onVaultUnlockSubmit: (password: string) => void
  pagerPageSize: number
}

/**
 * A `[[ … ]]` token sitting in the composer text, plus the payload it stands
 * for. `paste` tokens expand back into their text at submit; `image` tokens
 * are a receipt for a file the gateway already holds, and expand to nothing.
 *
 * `index` is the user-facing number in `[[ Image 2 ]]`; `path` is the gateway
 * path, used to detach the image when its token is deleted.
 */
export type ComposerToken =
  | { index: number; kind: 'image'; label: string; path: string; text?: undefined }
  | { index?: undefined; kind: 'paste'; label: string; path?: string; text: string }
