import { useEffect, useRef, useState } from 'react'

import type { CompletionItem } from '../app/interfaces.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { CompletionResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'

const TAB_PATH_RE = /((?:["']?(?:[A-Za-z]:[\\/]|\.{1,2}\/|~\/|\/|@|[^"'`\s]+\/))[^\s]*)$/

export function useCompletion(input: string, blocked: boolean, gw: GatewayClient) {
  const [completions, setCompletions] = useState<CompletionItem[]>([])
  const [compIdx, setCompIdx] = useState(0)
  const [compReplace, setCompReplace] = useState(0)
  const ref = useRef('')

  useEffect(() => {
    const clear = () => {
      setCompletions(prev => (prev.length ? [] : prev))
      setCompIdx(prev => (prev ? 0 : prev))
      setCompReplace(prev => (prev ? 0 : prev))
    }

    if (blocked) {
      ref.current = ''
      clear()

      return
    }

    if (input === ref.current) {
      return
    }

    ref.current = input

    const isSlash = input.startsWith('/')
    const pathWord = isSlash ? null : (input.match(TAB_PATH_RE)?.[1] ?? null)

    if (!isSlash && !pathWord) {
      clear()

      return
    }

    const pathReplace = input.length - (pathWord?.length ?? 0)

    const t = setTimeout(() => {
      if (ref.current !== input) {
        return
      }

      const req = isSlash
        ? gw.request<CompletionResponse>('complete.slash', { text: input })
        : gw.request<CompletionResponse>('complete.path', { word: pathWord })

      req
        .then(raw => {
          if (ref.current !== input) {
            return
          }

          const r = asRpcResult<CompletionResponse>(raw)

          setCompletions(r?.items ?? [])
          setCompIdx(0)
          setCompReplace(isSlash ? (r?.replace_from ?? 1) : pathReplace)
        })
        .catch((e: unknown) => {
          if (ref.current !== input) {
            return
          }

          setCompletions([
            {
              text: '',
              display: 'completion unavailable',
              meta: e instanceof Error && e.message ? e.message : 'unavailable'
            }
          ])
          setCompIdx(0)
          setCompReplace(isSlash ? 1 : pathReplace)
        })
    }, 60)

    return () => clearTimeout(t)
  }, [blocked, gw, input])

  return { completions, compIdx, setCompIdx, compReplace }
}
