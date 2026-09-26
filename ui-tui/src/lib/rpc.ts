import { getUiState } from '../app/uiStore.js'
import { describeRpcError } from '../app/userMessages.js'
import { translate } from '../i18n/index.js'

export type RpcResult = Record<string, any>

export const asRpcResult = <T extends RpcResult = RpcResult>(value: unknown): T | null =>
  !value || typeof value !== 'object' || Array.isArray(value) ? null : (value as T)

// Every `error: …` line the TUI prints for a failed RPC goes through here, so
// transport-level failures (backend down, stale session id, version skew,
// timeouts) read as what happened + what to do instead of the wire text.
export const rpcErrorMessage = (err: unknown) =>
  err instanceof Error && err.message
    ? describeRpcError(err, getUiState().locale)
    : typeof err === 'string' && err.trim()
      ? describeRpcError(err, getUiState().locale)
      : translate(getUiState().locale, 'messages.requestFailed')
