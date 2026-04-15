export type RpcResult = Record<string, any>

export const asRpcResult = <T extends RpcResult = RpcResult>(value: unknown): T | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  return value as T
}

export const rpcErrorMessage = (err: unknown) => {
  if (err instanceof Error && err.message) {
    return err.message
  }

  if (typeof err === 'string' && err.trim()) {
    return err
  }

  return 'request failed'
}
