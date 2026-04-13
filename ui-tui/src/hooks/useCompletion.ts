import { startTransition, useEffect, useRef, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'

const TAB_PATH_RE = /((?:["']?(?:[A-Za-z]:[\\/]|\.{1,2}\/|~\/|\/|@|[^"'`\s]+\/))[^\s]*)$/

export function useCompletion(input: string, blocked: boolean, gw: GatewayClient) {
  const [completions, setCompletions] = useState<{ text: string; display: string; meta: string }[]>([])
  const [compIdx, setCompIdx] = useState(0)
  const [compReplace, setCompReplace] = useState(0)
  const ref = useRef('')

  useEffect(() => {
    if (blocked) {
      if (completions.length) {
        setCompletions([])
        setCompIdx(0)
      }

      return
    }

    if (input === ref.current) {
      return
    }

    ref.current = input

    const isSlash = input.startsWith('/')
    const pathWord = !isSlash ? (input.match(TAB_PATH_RE)?.[1] ?? null) : null

    if (!isSlash && !pathWord) {
      if (completions.length) {
        setCompletions([])
        setCompIdx(0)
      }

      return
    }

    const t = setTimeout(() => {
      if (ref.current !== input) {
        return
      }

      const req = isSlash
        ? gw.request('complete.slash', { text: input })
        : gw.request('complete.path', { word: pathWord })

      req
        .then((r: any) => {
          if (ref.current !== input) {
            return
          }

          startTransition(() => {
            setCompletions(r?.items ?? [])
            setCompIdx(0)
            setCompReplace(isSlash ? (r?.replace_from ?? 1) : input.length - (pathWord?.length ?? 0))
          })
        })
        .catch((e: unknown) => {
          if (ref.current !== input) {
            return
          }

          const meta = e instanceof Error && e.message ? e.message : 'unavailable'
          startTransition(() => {
            setCompletions([{ text: '', display: 'completion unavailable', meta }])
            setCompIdx(0)
            setCompReplace(isSlash ? 1 : input.length - (pathWord?.length ?? 0))
          })
        })
    }, 60)

    return () => clearTimeout(t)
  }, [input, blocked, gw]) // eslint-disable-line react-hooks/exhaustive-deps

  return { completions, compIdx, setCompIdx, compReplace }
}
