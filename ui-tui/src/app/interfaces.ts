import type { ScrollBoxHandle } from '@hermes/ink'
import type { MutableRefObject, ReactNode, RefObject, SetStateAction } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { RpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import type {
  ActiveTool,
  ActivityItem,
  ApprovalReq,
  ClarifyReq,
  DetailsMode,
  Msg,
  PanelSection,
  SecretReq,
  SessionInfo,
  SlashCatalog,
  SudoReq,
  Usage
} from '../types.js'

export interface StateSetter<T> {
  (value: SetStateAction<T>): void
}

export interface SelectionApi {
  copySelection: () => string
}

export interface CompletionItem {
  display: string
  meta?: string
  text: string
}

export interface GatewayRpc {
  (method: string, params?: Record<string, unknown>): Promise<RpcResult | null>
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
  approval: ApprovalReq | null
  clarify: ClarifyReq | null
  modelPicker: boolean
  pager: PagerState | null
  picker: boolean
  secret: SecretReq | null
  sudo: SudoReq | null
}

export interface PagerState {
  lines: string[]
  offset: number
  title?: string
}

export interface ToolCompleteRibbon {
  label: string
  line: string
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
  info: SessionInfo | null
  sid: string | null
  status: string
  statusBar: boolean
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

export interface ComposerActions {
  clearIn: () => void
  dequeue: () => string | undefined
  enqueue: (text: string) => void
  handleTextPaste: (event: PasteEvent) => ComposerPasteResult | null
  openEditor: () => void
  pushHistory: (text: string) => void
  replaceQueue: (index: number, text: string) => void
  setCompIdx: StateSetter<number>
  setHistoryIdx: StateSetter<number | null>
  setInput: StateSetter<string>
  setInputBuf: StateSetter<string[]>
  setPasteSnips: StateSetter<PasteSnippet[]>
  setQueueEdit: (index: number | null) => void
  syncQueue: () => void
}

export interface ComposerRefs {
  historyDraftRef: MutableRefObject<string>
  historyRef: MutableRefObject<string[]>
  queueEditRef: MutableRefObject<number | null>
  queueRef: MutableRefObject<string[]>
  submitRef: MutableRefObject<(value: string) => void>
}

export interface ComposerState {
  compIdx: number
  compReplace: number
  completions: CompletionItem[]
  historyIdx: number | null
  input: string
  inputBuf: string[]
  pasteSnips: PasteSnippet[]
  queueEditIdx: number | null
  queuedDisplay: string[]
}

export interface UseComposerStateOptions {
  gw: GatewayClient
  onClipboardPaste: (quiet?: boolean) => Promise<void> | void
  submitRef: MutableRefObject<(value: string) => void>
}

export interface UseComposerStateResult {
  actions: ComposerActions
  refs: ComposerRefs
  state: ComposerState
}

export interface InterruptTurnOptions {
  appendMessage: (msg: Msg) => void
  gw: { request: (method: string, params?: Record<string, unknown>) => Promise<unknown> }
  sid: string
  sys: (text: string) => void
}

export interface TurnActions {
  clearReasoning: () => void
  endReasoningPhase: () => void
  idle: () => void
  interruptTurn: (options: InterruptTurnOptions) => void
  pruneTransient: () => void
  pulseReasoningStreaming: () => void
  pushActivity: (text: string, tone?: ActivityItem['tone'], replaceLabel?: string) => void
  pushTrail: (line: string) => void
  scheduleReasoning: () => void
  scheduleStreaming: () => void
  setActivity: StateSetter<ActivityItem[]>
  setReasoning: StateSetter<string>
  setReasoningTokens: StateSetter<number>
  setReasoningActive: StateSetter<boolean>
  setToolTokens: StateSetter<number>
  setReasoningStreaming: StateSetter<boolean>
  setStreaming: StateSetter<string>
  setTools: StateSetter<ActiveTool[]>
  setTurnTrail: StateSetter<string[]>
}

export interface TurnRefs {
  activeToolsRef: MutableRefObject<ActiveTool[]>
  bufRef: MutableRefObject<string>
  interruptedRef: MutableRefObject<boolean>
  lastStatusNoteRef: MutableRefObject<string>
  persistedToolLabelsRef: MutableRefObject<Set<string>>
  protocolWarnedRef: MutableRefObject<boolean>
  reasoningRef: MutableRefObject<string>
  reasoningStreamingTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  reasoningTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  statusTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  streamTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  toolTokenAccRef: MutableRefObject<number>
  toolCompleteRibbonRef: MutableRefObject<ToolCompleteRibbon | null>
  turnToolsRef: MutableRefObject<string[]>
}

export interface TurnState {
  activity: ActivityItem[]
  reasoning: string
  reasoningTokens: number
  reasoningActive: boolean
  reasoningStreaming: boolean
  streaming: string
  toolTokens: number
  tools: ActiveTool[]
  turnTrail: string[]
}

export interface UseTurnStateResult {
  actions: TurnActions
  refs: TurnRefs
  state: TurnState
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
    scrollRef: RefObject<ScrollBoxHandle | null>
    scrollWithSelection: (delta: number) => void
    selection: SelectionApi
    stdout?: NodeJS.WriteStream
  }
  turn: {
    actions: TurnActions
    refs: TurnRefs
  }
  voice: {
    recording: boolean
    setProcessing: StateSetter<boolean>
    setRecording: StateSetter<boolean>
  }
  wheelStep: number
}

export interface InputHandlerResult {
  pagerPageSize: number
}

export interface GatewayEventHandlerContext {
  composer: {
    dequeue: () => string | undefined
    queueEditRef: MutableRefObject<number | null>
    sendQueued: (text: string) => void
  }
  gateway: GatewayServices
  session: {
    STARTUP_RESUME_ID: string
    colsRef: MutableRefObject<number>
    newSession: (msg?: string) => void
    resetSession: () => void
    setCatalog: StateSetter<SlashCatalog | null>
  }
  system: {
    bellOnComplete: boolean
    stdout?: NodeJS.WriteStream
    sys: (text: string) => void
  }
  transcript: {
    appendMessage: (msg: Msg) => void
    setHistoryItems: StateSetter<Msg[]>
    setMessages: StateSetter<Msg[]>
  }
  turn: {
    actions: Pick<
      TurnActions,
      | 'clearReasoning'
      | 'endReasoningPhase'
      | 'idle'
      | 'pruneTransient'
      | 'pulseReasoningStreaming'
      | 'pushActivity'
      | 'pushTrail'
      | 'scheduleReasoning'
      | 'scheduleStreaming'
      | 'setActivity'
      | 'setReasoningTokens'
      | 'setStreaming'
      | 'setToolTokens'
      | 'setTools'
      | 'setTurnTrail'
    >
    refs: Pick<
      TurnRefs,
      | 'activeToolsRef'
      | 'bufRef'
      | 'interruptedRef'
      | 'lastStatusNoteRef'
      | 'persistedToolLabelsRef'
      | 'protocolWarnedRef'
      | 'reasoningRef'
      | 'statusTimerRef'
      | 'toolTokenAccRef'
      | 'toolCompleteRibbonRef'
      | 'turnToolsRef'
    >
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
    catalog: SlashCatalog | null
    lastUserMsg: string
    maybeWarn: (value: any) => void
    messages: Msg[]
  }
  session: {
    closeSession: (targetSid?: string | null) => Promise<unknown>
    die: () => void
    guardBusySessionSwitch: (what?: string) => boolean
    newSession: (msg?: string) => void
    resetVisibleHistory: (info?: SessionInfo | null) => void
    resumeById: (id: string) => void
    setSessionStartedAt: StateSetter<number>
  }
  transcript: {
    page: (text: string, title?: string) => void
    panel: (title: string, sections: PanelSection[]) => void
    send: (text: string) => void
    setHistoryItems: StateSetter<Msg[]>
    setMessages: StateSetter<Msg[]>
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
  handleTextPaste: (event: PasteEvent) => ComposerPasteResult | null
  input: string
  inputBuf: string[]
  pagerPageSize: number
  queueEditIdx: number | null
  queuedDisplay: string[]
  submit: (value: string) => void
  updateInput: StateSetter<string>
}

export interface AppLayoutProgressProps {
  activity: ActivityItem[]
  reasoning: string
  reasoningTokens: number
  reasoningActive: boolean
  reasoningStreaming: boolean
  showProgressArea: boolean
  showStreamingArea: boolean
  streaming: string
  toolTokens: number
  tools: ActiveTool[]
  turnTrail: string[]
}

export interface AppLayoutStatusProps {
  cwdLabel: string
  durationLabel: string
  showStickyPrompt: boolean
  statusColor: string
  stickyPrompt: string
  voiceLabel: string
}

export interface AppLayoutTranscriptProps {
  historyItems: Msg[]
  scrollRef: RefObject<ScrollBoxHandle | null>
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
  text: string
}
