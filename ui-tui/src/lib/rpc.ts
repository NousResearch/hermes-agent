export type RpcResult = Record<string, any>

export const asRpcResult = <T extends RpcResult = RpcResult>(value: unknown): T | null =>
  !value || typeof value !== 'object' || Array.isArray(value) ? null : (value as T)

export const rpcErrorMessage = (err: unknown) =>
  err instanceof Error && err.message ? err.message : typeof err === 'string' && err.trim() ? err : 'request failed'
