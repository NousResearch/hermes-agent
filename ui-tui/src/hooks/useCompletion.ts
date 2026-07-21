import { useEffect, useRef, useState } from 'react'

import type { CompletionItem } from '../app/interfaces.js'
import { looksLikeSlashCommand } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { CompletionResponse } from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import type { SlashCatalog } from '../types.js'

const TAB_PATH_RE = /((?:["']?(?:[A-Za-z]:[\\/]|\.{1,2}\/|~\/|\/|@|[^"'`\s]+\/))[^\s]*)$/
const DYNAMIC_SLASH_COMMANDS = new Set(['/model', '/personality', '/skin'])

const completionText = (cmdName: string, word: string) => (cmdName === word ? `${cmdName} ` : cmdName)

interface LocalSlashEntry {
  display: string
  meta: string
  name: string
}

const buildLocalSlashEntries = (catalog: SlashCatalog): LocalSlashEntry[] => {
  const aliasesByCanonical = new Map<string, string[]>()

  for (const [raw, canonical] of Object.entries(catalog.canon)) {
    const name = raw.toLowerCase()
    const canonicalName = canonical.toLowerCase()

    if (name === canonicalName) {
      continue
    }

    const aliases = aliasesByCanonical.get(canonicalName) ?? []
    aliases.push(name)
    aliasesByCanonical.set(canonicalName, aliases)
  }

  const entries: LocalSlashEntry[] = []

  for (const [command, description] of catalog.pairs) {
    const display = command.toLowerCase()
    const meta = String(description)

    entries.push({ display, meta, name: display })

    for (const alias of aliasesByCanonical.get(display) ?? []) {
      entries.push({
        display: alias,
        meta: `${meta} (alias for ${display})`,
        name: alias
      })
    }
  }

  return entries
}

export function getLocalSlashCompletion(text: string, catalog: null | SlashCatalog): CompletionResponse | null {
  if (!catalog || !looksLikeSlashCommand(text)) {
    return null
  }

  const entries = buildLocalSlashEntries(catalog)
  const localNames = new Set(entries.map(entry => entry.name))
  const parts = text.split(/\s+/, 2)
  const baseInput = (parts[0] ?? '').toLowerCase()
  const baseCanonical = (catalog.canon[baseInput] ?? baseInput).toLowerCase()
  const replaceFrom = text.includes(' ') ? text.lastIndexOf(' ') + 1 : 1

  if (parts.length > 1 || text.endsWith(' ')) {
    if (DYNAMIC_SLASH_COMMANDS.has(baseCanonical)) {
      return null
    }

    const subText = parts[1] ?? ''

    if (!subText.includes(' ')) {
      const subs = catalog.sub[baseCanonical] ?? []

      if (subs.length) {
        return {
          items: subs
            .filter(sub => sub.startsWith(subText.toLowerCase()) && sub !== subText.toLowerCase())
            .slice(0, 30)
            .map(sub => ({ display: sub, text: sub })),
          replace_from: replaceFrom
        }
      }
    }

    if (localNames.has(baseInput) || localNames.has(baseCanonical)) {
      return { items: [], replace_from: replaceFrom }
    }

    return null
  }

  const word = text.slice(1).toLowerCase()

  const items = entries
    .filter(entry => entry.name.slice(1).startsWith(word))
    .slice(0, 30)
    .map(entry => ({
      display: entry.display,
      meta: entry.meta,
      text: completionText(entry.name.slice(1), word)
    }))

  return items.length ? { items, replace_from: 1 } : null
}

export function completionRequestForInput(
  input: string
):
  | { method: 'complete.path'; params: { word: string }; replaceFrom: number }
  | { method: 'complete.slash'; params: { text: string }; replaceFrom: number }
  | null {
  const isSlashCommand = looksLikeSlashCommand(input)
  const pathWord = isSlashCommand ? null : (input.match(TAB_PATH_RE)?.[1] ?? null)

  if (!isSlashCommand && !pathWord) {
    return null
  }

  // `/model` uses the two-step ModelPicker (real curated IDs).
  // Slash completion here only showed short aliases + vendor/family meta.
  if (isSlashCommand && /^\/model(?:\s|$)/.test(input)) {
    return null
  }

  if (isSlashCommand) {
    return { method: 'complete.slash', params: { text: input }, replaceFrom: 1 }
  }

  return {
    method: 'complete.path',
    params: { word: pathWord! },
    replaceFrom: input.length - pathWord!.length
  }
}

export function useCompletion(input: string, blocked: boolean, gw: GatewayClient, catalog: null | SlashCatalog) {
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

      const localSlash = request.method === 'complete.slash' ? getLocalSlashCompletion(input, catalog) : null

      if (localSlash) {
        setCompletions(localSlash.items ?? [])
        setCompIdx(0)
        setCompReplace(localSlash.replace_from ?? 1)

        return
      }

      gw.request<CompletionResponse>(request.method, request.params)
        .then(raw => {
          if (ref.current !== input) {
            return
          }

          const r = asRpcResult<CompletionResponse>(raw)

          setCompletions(r?.items ?? [])
          setCompIdx(0)
          setCompReplace(request.method === 'complete.slash' ? (r?.replace_from ?? 1) : request.replaceFrom)
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
  }, [blocked, catalog, gw, input])

  return { completions, compIdx, setCompIdx, compReplace }
}
