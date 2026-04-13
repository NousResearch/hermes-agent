export type RpcResult = Record<string, any>

export const asRpcResult = (value: unknown): RpcResult | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  return value as RpcResult
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
