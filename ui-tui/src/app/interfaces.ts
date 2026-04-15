import type { GatewayClient } from '../gatewayClient.js'
import type { Theme } from '../theme.js'
import type { ApprovalReq, ClarifyReq, DetailsMode, Msg, SecretReq, SessionInfo, SudoReq, Usage } from '../types.js'

export interface CompletionItem {
  display: string
  meta?: string
  text: string
}

export interface GatewayRpc {
  (method: string, params?: Record<string, unknown>): Promise<any | null>
}

export interface GatewayServices {
  gw: GatewayClient
  rpc: GatewayRpc
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
