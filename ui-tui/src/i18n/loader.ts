// Fetch the TUI language from the backend. The locale id comes from
// `display.language` (already read by useConfigSync's `config.get full`); the
// strings come from `i18n.catalog {lang, surface: 'tui'}`. A backend that
// predates `i18n.catalog` (method not found) or has no pack for the language
// leaves the UI in English under that locale id.

import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult } from '../lib/rpc.js'

import { applyLocale, DEFAULT_LOCALE, getLocale } from './runtime.js'
import type { CatalogPack } from './types.js'

export const TUI_SURFACE = 'tui'

/** `display.language` normalization shared with the backend's `_normalize_lang`:
 *  lowercase, `_` → `-`, blank → en. Unknown ids are kept: the pack decides. */
export function normalizeLanguageId(raw: unknown): string {
  if (typeof raw !== 'string') {
    return DEFAULT_LOCALE
  }

  const id = raw.trim().toLowerCase().replace(/_/g, '-')

  return id || DEFAULT_LOCALE
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

let inFlight: Promise<void> | null = null
let lastRequested = ''

/** Apply `display.language`: no-op when it did not change; English needs no
 *  RPC. Coalesces concurrent calls (config poll + boot hydration). */
export function syncTuiLocale(gw: Pick<GatewayClient, 'request'>, rawLanguage: unknown): Promise<void> {
  const lang = normalizeLanguageId(rawLanguage)

  if (lang === lastRequested && (inFlight || lang === getLocale())) {
    return inFlight ?? Promise.resolve()
  }

  lastRequested = lang

  if (lang === DEFAULT_LOCALE) {
    applyLocale(DEFAULT_LOCALE, null)
    inFlight = null

    return Promise.resolve()
  }

  const run = fetchCatalogPack((method, params) => gw.request(method, params), lang).then(pack => {
    // A newer request superseded this one while it was in flight.
    if (lastRequested === lang) {
      applyLocale(lang, pack)
    }
  })

  inFlight = run.finally(() => {
    if (inFlight === run) {
      inFlight = null
    }
  })

  return inFlight
}

/** Test seam: forget the last requested language. */
export function resetTuiLocaleSync(): void {
  inFlight = null
  lastRequested = ''
}
