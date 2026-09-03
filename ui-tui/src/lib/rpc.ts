import type { CommandDispatchResponse } from '../gatewayTypes.js'

export type RpcResult = Record<string, any>

export const asRpcResult = <T extends RpcResult = RpcResult>(value: unknown): T | null =>
  !value || typeof value !== 'object' || Array.isArray(value) ? null : (value as T)

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

  const str = (value: unknown) => (typeof value === 'string' ? value : undefined)

  if (t === 'skill' && typeof o.name === 'string') {
    return { type: 'skill', name: o.name, message: str(o.message), display: str(o.display) }
  }

  if (t === 'send' && typeof o.message === 'string') {
    return {
      type: 'send',
      message: o.message,
      notice: str(o.notice),
      display: str(o.display)
    }
  }

  if (t === 'prefill' && typeof o.message === 'string') {
    return {
      type: 'prefill',
      message: o.message,
      notice: typeof o.notice === 'string' ? o.notice : undefined
    }
  }

  return null
}

export const rpcErrorMessage = (err: unknown) =>
  err instanceof Error && err.message ? err.message : typeof err === 'string' && err.trim() ? err : 'request failed'
