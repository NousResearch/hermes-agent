import { useEffect, useMemo, useRef, useState } from 'react'

import type { CompletionItem } from '../app/interfaces.js'
import { looksLikeSlashCommand } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { CompletionResponse, GatewayCompletionItem } from '../gatewayTypes.js'
import { translate, translateSlashDescription, type TranslationKey, useI18n } from '../i18n/index.js'
import { asRpcResult } from '../lib/rpc.js'
import { listWidgetApps, widgetHelp } from '../sdk/registry.js'

/** Client-side widget apps live in the TUI's registry, not the gateway — so
 *  `/` completions merge their title/metadata here. Registry-driven: a new
 *  app surfaces automatically, no hardcoded lists on either side. */
export function mergeWidgetAppItems(
  input: string,
  items: CompletionItem[],
  locale: Parameters<typeof translate>[0] = 'en'
): CompletionItem[] {
  // Only complete the command NAME position (no args typed yet).
  if (input.includes(' ')) {
    return items
  }

  const local = listWidgetApps()
    .filter(app => `/${app.id}`.startsWith(input.toLowerCase()))
    .filter(app => !items.some(item => item.text === `/${app.id}`))
    .map(app => ({ display: `/${app.id}`, meta: widgetHelp(app, locale), text: `/${app.id}` }))

  return [...items, ...local]
}

const TAB_PATH_RE = /((?:["']?(?:[A-Za-z]:[\\/]|\.{1,2}\/|~\/|\/|@|[^"'`\s]+\/))[^\s]*)$/

export interface LocalizableCompletionItem extends CompletionItem {
  displayTranslationKey?: TranslationKey
  metaTranslationKey?: TranslationKey
  metaTranslationVars?: Record<string, string | number>
  slashDescriptionId?: string
}

export const localizeCompletionItems = (
  items: readonly LocalizableCompletionItem[],
  locale: Parameters<typeof translate>[0]
): CompletionItem[] =>
  items.map(item => ({
    display: item.displayTranslationKey ? translate(locale, item.displayTranslationKey) : item.display,
    meta: item.slashDescriptionId
      ? translateSlashDescription(locale, item.slashDescriptionId, item.meta ?? '')
      : item.metaTranslationKey
        ? translate(locale, item.metaTranslationKey, item.metaTranslationVars)
        : item.meta,
    text: item.text
  }))

export const localizableCompletionItem = (item: GatewayCompletionItem): LocalizableCompletionItem => {
  const presentationKey = item.meta_key?.startsWith('completion.') ? (item.meta_key as TranslationKey) : undefined

  return {
    display: item.display,
    meta: item.meta,
    metaTranslationKey: presentationKey,
    metaTranslationVars: item.meta_vars,
    slashDescriptionId: presentationKey ? undefined : item.meta_key,
    text: item.text
  }
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

export function useCompletion(input: string, blocked: boolean, gw: GatewayClient) {
  const { locale } = useI18n()
  const [rawCompletions, setRawCompletions] = useState<LocalizableCompletionItem[]>([])
  const [compIdx, setCompIdx] = useState(0)
  const [compReplace, setCompReplace] = useState(0)
  const ref = useRef('')

  const completions = useMemo<CompletionItem[]>(
    () => localizeCompletionItems(rawCompletions, locale),
    [locale, rawCompletions]
  )

  useEffect(() => {
    const clear = () => {
      setRawCompletions(prev => (prev.length ? [] : prev))
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

          const items =
            request.method === 'complete.slash' ? mergeWidgetAppItems(input, r?.items ?? [], locale) : (r?.items ?? [])

          setRawCompletions(items.map(localizableCompletionItem))
          setCompIdx(0)
          setCompReplace(request.method === 'complete.slash' ? (r?.replace_from ?? 1) : request.replaceFrom)
        })
        .catch((e: unknown) => {
          if (ref.current !== input) {
            return
          }

          setRawCompletions([
            {
              text: '',
              display: '',
              displayTranslationKey: 'completion.unavailable',
              meta: e instanceof Error && e.message ? e.message : undefined,
              metaTranslationKey: e instanceof Error && e.message ? undefined : 'completion.unavailableMeta'
            }
          ])
          setCompIdx(0)
          setCompReplace(request.replaceFrom)
        })
    }, 60)

    return () => clearTimeout(t)
  }, [blocked, gw, input, locale])

  return { completions, compIdx, setCompIdx, compReplace }
}
