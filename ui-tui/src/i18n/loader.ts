import { normalizeLanguageIdentity } from '@hermes/shared/locale-registry'
// Fetch the TUI language from the backend. The locale id comes from
// `display.language` (already read by useConfigSync's `config.get full`); the
// strings come from `i18n.catalog {lang, surface: 'tui'}`. A backend that
// predates `i18n.catalog` (method not found) or has no pack for the language
// leaves the UI in English under that locale id.

import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult } from '../lib/rpc.js'

import { applyLocale, DEFAULT_LOCALE } from './runtime.js'
import type { CatalogPack } from './types.js'

export const TUI_SURFACE = 'tui'

/** `display.language` normalization shared with the backend's `_normalize_lang`:
 *  lowercase, `_` → `-`, blank → en. Unknown ids are kept: the pack decides. */
export function normalizeLanguageId(raw: unknown): string {
  if (typeof raw !== 'string') {
    return DEFAULT_LOCALE
  }

  const id = raw.trim().toLowerCase().replace(/_/g, '-')

  return normalizeLanguageIdentity(id)
}

type RequestFn = <T>(method: string, params: Record<string, unknown>) => Promise<T>

export async function fetchCatalogPack(request: RequestFn, lang: string): Promise<CatalogPack | null> {
  try {
    const raw = asRpcResult<Partial<CatalogPack>>(await request('i18n.catalog', { lang, surface: TUI_SURFACE }))

    if (!raw || typeof raw.messages !== 'object' || raw.messages === null) {
      return null
    }

    return { lang: typeof raw.lang === 'string' ? raw.lang : lang, messages: raw.messages, surface: TUI_SURFACE }
  } catch {
    // Method not found / transport failure: English stays.
    return null
  }
}

interface LocaleRequest {
  gw: Pick<GatewayClient, 'request'>
  lang: string
  signal?: AbortSignal
  promise: Promise<boolean>
}
let current: LocaleRequest | null = null

/** Coalesce one active scope's requests. A failed fetch remains retryable, and
 * a late response cannot replace a newer session's catalog. */
export function syncTuiLocale(
  gw: Pick<GatewayClient, 'request'>,
  rawLanguage: unknown,
  signal?: AbortSignal
): Promise<boolean> {
  if (signal?.aborted) {
    return Promise.resolve(false)
  }
  const lang = normalizeLanguageId(rawLanguage)

  if (current?.gw === gw && current.lang === lang && current.signal === signal) {
    return current.promise
  }

  const request: LocaleRequest = { gw, lang, signal, promise: Promise.resolve(true) }
  current = request

  if (lang === DEFAULT_LOCALE) {
    applyLocale(lang, null)

    return request.promise
  }

  request.promise = fetchCatalogPack((method, params) => gw.request(method, params), lang).then(pack => {
    if (current !== request || signal?.aborted) {
      return false
    }
    applyLocale(pack?.lang ?? lang, pack)

    if (!pack) {
      current = null
    }

    return pack !== null
  })

  return request.promise
}

export function resetTuiLocaleSync(): void {
  current = null
}
