import { useEffect, useRef, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'

const TAB_PATH_RE = /((?:["']?(?:[A-Za-z]:[\\/]|\.{1,2}\/|~\/|\/|@|[^"'`\s]+\/))[^\s]*)$/

export function useCompletion(input: string, blocked: boolean, gw: GatewayClient) {
  const [completions, setCompletions] = useState<{ text: string; display: string; meta: string }[]>([])
  const [compIdx, setCompIdx] = useState(0)
  const [compReplace, setCompReplace] = useState(0)
  const ref = useRef('')

  useEffect(() => {
    const clear = () => {
      if (!completions.length) {
        return
      }

      setCompletions([])
      setCompIdx(0)
    }

    if (blocked || input === ref.current) {
      if (blocked) {
        clear()
      }

      return
    }

    ref.current = input

    const isSlash = input.startsWith('/')
    const pathWord = !isSlash ? (input.match(TAB_PATH_RE)?.[1] ?? null) : null

    if (!isSlash && !pathWord) {
      clear()

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

          setCompletions(r?.items ?? [])
          setCompIdx(0)
          setCompReplace(isSlash ? (r?.replace_from ?? 1) : input.length - (pathWord?.length ?? 0))
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
          setCompReplace(isSlash ? 1 : input.length - (pathWord?.length ?? 0))
        })
    }, 60)

    return () => clearTimeout(t)
  }, [input, blocked, gw]) // eslint-disable-line react-hooks/exhaustive-deps

  return { completions, compIdx, setCompIdx, compReplace }
}
