import { useEffect, useRef, useState } from 'react'

import type { CompletionItem } from '../app/interfaces.js'
import { rankSlashItems } from '../app/slash/fuzzyScore.js'
import { inlineSlashTrigger, looksLikeSlashCommand } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { CompletionResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import { listWidgetApps } from '../sdk/registry.js'

/** Client-side widget apps live in the TUI's registry, not the gateway — so
 *  `/` completions merge their title/metadata here. Registry-driven: a new
 *  app surfaces automatically, no hardcoded lists on either side. Matching is
 *  description-aware (ported from grok-cli's slash menu): `/timer` surfaces a
 *  widget whose help text mentions timers, not just id-prefix hits. */
export function mergeWidgetAppItems(input: string, items: CompletionItem[]): CompletionItem[] {
  // Only complete the command NAME position (no args typed yet).
  if (input.includes(' ')) {
    return items
  }

  const local = rankSlashItems(listWidgetApps(), input, app => ({ description: app.help, id: app.id }))
    .filter(app => !items.some(item => item.text === `/${app.id}`))
    .map(app => ({ display: `/${app.id}`, meta: app.help, text: `/${app.id}` }))

  return [...items, ...local]
}

const TAB_PATH_RE = /((?:["']?(?:[A-Za-z]:[\\/]|\.{1,2}\/|~\/|\/|@|[^"'`\s]+\/))[^\s]*)$/

export function completionRequestForInput(
  input: string
):
  | { method: 'complete.path'; params: { word: string }; replaceFrom: number }
  | { method: 'complete.slash'; params: { text: string }; replaceFrom: number; skillsOnly?: boolean }
  | null {
  const isSlashCommand = looksLikeSlashCommand(input)
  const pathWord = isSlashCommand ? null : (input.match(TAB_PATH_RE)?.[1] ?? null)

  // `/model` uses the two-step ModelPicker (real curated IDs).
  // Slash completion here only showed short aliases + vendor/family meta.
  if (isSlashCommand && /^\/model(?:\s|$)/.test(input)) {
    return null
  }

  // A `/token` mid-message is a skill reference dropped into prose. Detected
  // BEFORE the leading-command shape because only the first slash can be an
  // invocation — `/help /cle` is a command whose argument names a skill, and
  // routing the whole line to the backend's completer offered nothing at all.
  // It only matches a whitespace-preceded slash sitting at the caret, so
  // ordinary argument completion (`/cron ad`, `/personality alic`) is
  // untouched.
  const inline = inlineSlashTrigger(input)

  if (inline) {
    return {
      method: 'complete.slash',
      params: { text: `/${inline.query}` },
      replaceFrom: inline.start + 1,
      skillsOnly: true
    }
  }

  if (isSlashCommand) {
    return { method: 'complete.slash', params: { text: input }, replaceFrom: 1 }
  }

  if (!pathWord) {
    return null
  }

  return {
    method: 'complete.path',
    params: { word: pathWord },
    replaceFrom: input.length - pathWord.length
  }
}

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

    const request = completionRequestForInput(input)

    if (!request) {
      clear()

      return
    }

    const t = setTimeout(() => {
      if (ref.current !== input) {
        return
      }

      gw.request<CompletionResponse>(request.method, request.params)
        .then(raw => {
          if (ref.current !== input) {
            return
          }

          const r = asRpcResult<CompletionResponse>(raw)

          const fetched =
            request.method === 'complete.slash' ? mergeWidgetAppItems(input, r?.items ?? []) : (r?.items ?? [])

          // Mid-message offers SKILLS only. A built-in like `/model` or `/new`
          // acts on the app, so it's meaningless as a reference inside prose —
          // only a skill reads as "handle this part with X". Filtering here
          // rather than in the gateway keeps one completion source for both
          // shapes.
          const items =
            request.method === 'complete.slash' && request.skillsOnly
              ? fetched.filter(item => item.kind === 'skill')
              : fetched

          setCompletions(items)
          setCompIdx(0)
          // An inline reference replaces its own token, so the gateway's
          // `replace_from` (an offset into the synthetic `/query` it was sent)
          // doesn't apply — the caller already knows where the token starts.
          setCompReplace(
            request.method === 'complete.slash' && !request.skillsOnly ? (r?.replace_from ?? 1) : request.replaceFrom
          )
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
          setCompReplace(request.replaceFrom)
        })
    }, 60)

    return () => clearTimeout(t)
  }, [blocked, gw, input])

  return { completions, compIdx, setCompIdx, compReplace }
}
