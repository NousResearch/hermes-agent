import type { CommandDispatchResponse } from '../gatewayTypes.js'

export type RpcResult = Record<string, any>

export const asRpcResult = <T extends RpcResult = RpcResult>(value: unknown): T | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  return value as T
}

export const asCommandDispatch = (value: unknown): CommandDispatchResponse | null => {
  const o = asRpcResult(value)

  if (!o || typeof o.type !== 'string') {
    return null
  }

  const t = o.type

  if (t === 'exec' || t === 'plugin') {
    return { type: t, output: typeof o.output === 'string' ? o.output : undefined }
  }

  if (t === 'alias' && typeof o.target === 'string') {
    return { type: 'alias', target: o.target }
  }

  if (t === 'skill' && typeof o.name === 'string') {
    return {
      type: 'skill',
      name: o.name,
      message: typeof o.message === 'string' ? o.message : undefined
    }
  }

  return null
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
