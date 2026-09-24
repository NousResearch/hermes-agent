import { describeRpcError } from '../app/userMessages.js'

export type RpcResult = Record<string, any>

export const asRpcResult = <T extends RpcResult = RpcResult>(value: unknown): T | null =>
  !value || typeof value !== 'object' || Array.isArray(value) ? null : (value as T)

// Every `error: …` line the TUI prints for a failed RPC goes through here, so
// transport-level failures (backend down, stale session id, version skew,
// timeouts) read as what happened + what to do instead of the wire text.
export const rpcErrorMessage = (err: unknown) =>
  err instanceof Error && err.message
    ? describeRpcError(err)
    : typeof err === 'string' && err.trim()
      ? describeRpcError(err)
      : 'request failed'
