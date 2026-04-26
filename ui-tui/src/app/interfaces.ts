import type { ScrollBoxHandle } from '@hermes/ink'
import type { MutableRefObject, ReactNode, RefObject, SetStateAction } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ImageAttachResponse } from '../gatewayTypes.js'
import type { RpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import type {
  ActiveTool,
  ActivityItem,
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
  SubagentProgress,
  SudoReq,
  Usage
} from '../types.js'

export interface StateSetter<T> {
  (value: SetStateAction<T>): void
}

export type StatusBarMode = 'bottom' | 'off' | 'top'

export interface SelectionApi {
  captureScrolledRows: (firstRow: number, lastRow: number, side: 'above' | 'below') => void
  clearSelection: () => void
  copySelection: () => string
  getState: () => unknown
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

export interface OverlayState {
  agents: boolean
  agentsInitialHistoryIndex: number
  approval: ApprovalReq | null
  clarify: ClarifyReq | null
  confirm: ConfirmReq | null
  modelPicker: boolean
  pager: null | PagerState
  picker: boolean
  secret: null | SecretReq
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
  bgTasks: Set<string>
  busy: boolean
  compact: boolean
  detailsMode: DetailsMode
  info: null | SessionInfo
  inlineDiffs: boolean
  mouseTracking: boolean
  sections: SectionVisibility
  showCost: boolean
  showReasoning: boolean
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
  newSession: (msg?: string) => void
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
    recording: boolean
    setProcessing: StateSetter<boolean>
    setRecording: StateSetter<boolean>
    setVoiceEnabled: StateSetter<boolean>
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
    newSession: (msg?: string) => void
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
  }
}

export interface SlashHandlerContext {
  composer: {
    enqueue: (text: string) => void
    hasSelection: boolean
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
  }
  session: {
    closeSession: (targetSid?: null | string) => Promise<unknown>
    die: () => void
    guardBusySessionSwitch: (what?: string) => boolean
    newSession: (msg?: string) => void
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
  }
}

export interface AppLayoutActions {
  answerApproval: (choice: string) => void
  answerClarify: (answer: string) => void
  answerSecret: (value: string) => void
  answerSudo: (pw: string) => void
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
}

export interface AppLayoutProgressProps {
  activity: ActivityItem[]
  outcome: string
  reasoning: string
  reasoningActive: boolean
  reasoningStreaming: boolean
  reasoningTokens: number
  showProgressArea: boolean
  showStreamingArea: boolean
  streamPendingTools: string[]
  streamSegments: Msg[]
  streaming: string
  subagents: SubagentProgress[]
  toolTokens: number
  tools: ActiveTool[]
  turnTrail: string[]
}

export interface AppLayoutStatusProps {
  cwdLabel: string
  goodVibesTick: number
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
  mouseTracking: boolean
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
  onModelSelect: (value: string) => void
  onPickerSelect: (sessionId: string) => void
  onSecretSubmit: (value: string) => void
  onSudoSubmit: (pw: string) => void
  pagerPageSize: number
}

export interface PasteSnippet {
  label: string
  path?: string
  text: string
}
