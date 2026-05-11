import { useEffect, useMemo, useState } from 'react'

const titleCache = new Map<string, string>()
const titleInflight = new Map<string, Promise<string>>()
const titleSubs = new Map<string, Set<(value: string) => void>>()

const TITLE_CACHE_LIMIT = 500
const TITLE_BYTE_BUDGET = 96 * 1024
const TITLE_TIMEOUT_MS = 5000

const TITLE_USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'

const TITLE_ERROR_RE =
  /\b(?:access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i

const DOMAIN_RE = /^(?:www\.)?[a-z0-9](?:[a-z0-9-]*\.)+[a-z]{2,}(?::\d+)?(?:[/?#][^\s]*)?$/i
const SKIP_PROTO_RE = /^(?:file|data|mailto|javascript|blob|chrome|about|hermes):/i
const LOCAL_HOST_RE = /^(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?$/i

const HTML_ENTITIES: Record<string, string> = {
  '#39': "'",
  amp: '&',
  apos: "'",
  gt: '>',
  lt: '<',
  nbsp: ' ',
  quot: '"'
}

export function normalizeExternalUrl(value: string): string {
  const trimmed = value.trim()

  if (!trimmed || /^https?:\/\//i.test(trimmed)) {
    return trimmed
  }

  return DOMAIN_RE.test(trimmed) ? `https://${trimmed}` : trimmed
}

function parseUrl(value: string): null | URL {
  try {
    return new URL(normalizeExternalUrl(value))
  } catch {
    return null
  }
}

function titleCacheKey(value: string): string {
  const url = parseUrl(value)

  if (!url) {
    return normalizeExternalUrl(value)
  }

  const host = url.hostname.replace(/^www\./i, '').toLowerCase()
  const pathname = url.pathname === '/' ? '/' : url.pathname.replace(/\/+$/, '') || '/'

  return `${host}${pathname}${url.search || ''}`
}

function cacheTitle(key: string, title: string): void {
  if (titleCache.size >= TITLE_CACHE_LIMIT) {
    titleCache.delete(titleCache.keys().next().value as string)
  }

  titleCache.set(key, title)
}

export function hostPathLabel(value: string): string {
  const url = parseUrl(value)

  if (!url) {
    return value
  }

  const host = url.hostname.replace(/^www\./, '')
  const path = url.pathname && url.pathname !== '/' ? url.pathname.replace(/\/$/, '') : ''

  return `${host}${path}`
}

function cleanSlug(segment: string): string {
  try {
    return decodeURIComponent(segment)
      .replace(/\.a\d+\..*$/i, '')
      .replace(/\.(?:html?|php|aspx?)$/i, '')
      .replace(/(?:[-_.](?:[a-z]{1,3}\d{2,}|i\d{2,}))+$/i, '')
      .replace(/[_-]+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  } catch {
    return ''
  }
}

export function urlSlugTitleLabel(value: string): string {
  const url = parseUrl(value)

  for (const segment of url?.pathname.split('/').filter(Boolean).reverse() ?? []) {
    const cleaned = cleanSlug(segment)

    if (!cleaned || !/[a-z]/i.test(cleaned)) {
      continue
    }

    if (/^(?:[a-z]{1,3}\d+|\d+)$/i.test(cleaned.replace(/\s+/g, ''))) {
      continue
    }

    const titled = cleaned.replace(/\b[a-z]/g, c => c.toUpperCase())

    if (titled.length >= 4) {
      return titled
    }
  }

  return hostPathLabel(value)
}

export function isTitleFetchable(value: string): boolean {
  if (!value || SKIP_PROTO_RE.test(value)) {
    return false
  }

  const url = parseUrl(value)

  return Boolean(url && /^https?:$/.test(url.protocol) && !LOCAL_HOST_RE.test(url.host))
}

function decodeHtmlEntities(value: string): string {
  return value
    .replace(/&(amp|lt|gt|quot|apos|nbsp|#39);/gi, (_match, key: string) => HTML_ENTITIES[key.toLowerCase()] ?? '')
    .replace(/&#x([0-9a-f]+);/gi, (_match, hex: string) => String.fromCodePoint(parseInt(hex, 16) || 32))
    .replace(/&#(\d+);/g, (_match, decimal: string) => String.fromCodePoint(parseInt(decimal, 10) || 32))
}

function parseHtmlTitle(html: string): string {
  const raw = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]

  return raw ? decodeHtmlEntities(raw).replace(/\s+/g, ' ').trim() : ''
}

async function readResponseSnippet(response: Response): Promise<string> {
  const reader = response.body?.getReader()

  if (!reader) {
    return (await response.text()).slice(0, TITLE_BYTE_BUDGET)
  }

  const chunks: Uint8Array[] = []
  let done = false
  let bytes = 0

  try {
    while (bytes < TITLE_BYTE_BUDGET) {
      const chunk = await reader.read()

      if (chunk.done) {
        done = true

        break
      }

      const value = chunk.value

      if (!value?.length) {
        continue
      }

      const remaining = TITLE_BYTE_BUDGET - bytes
      const next = value.length > remaining ? value.subarray(0, remaining) : value

      chunks.push(next)
      bytes += next.length

      if (next.length < value.length) {
        break
      }
    }
  } catch {
    return ''
  } finally {
    if (!done) {
      try {
        await reader.cancel()
      } catch {
        // Ignore stream teardown failures.
      }
    }
  }

  if (!chunks.length) {
    return ''
  }

  const joined = new Uint8Array(bytes)
  let offset = 0

  for (const chunk of chunks) {
    joined.set(chunk, offset)
    offset += chunk.length
  }

  return new TextDecoder().decode(joined)
}

function usableTitle(value: string): string {
  const clean = value.replace(/\s+/g, ' ').trim()

  return clean && !TITLE_ERROR_RE.test(clean) ? clean : ''
}

async function fetchHtmlTitle(normalizedUrl: string): Promise<string> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), TITLE_TIMEOUT_MS)

  try {
    const response = await fetch(normalizedUrl, {
      headers: {
        Accept: 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.5',
        'Accept-Language': 'en-US,en;q=0.7',
        'User-Agent': TITLE_USER_AGENT
      },
      redirect: 'follow',
      signal: controller.signal
    })

    if (!response.ok) {
      return ''
    }

    const contentType = response.headers.get('content-type')

    if (contentType && !/(?:html|xml|text\/html)/i.test(contentType)) {
      return ''
    }

    const html = await readResponseSnippet(response)

    return parseHtmlTitle(html).slice(0, 240)
  } catch {
    return ''
  } finally {
    clearTimeout(timeout)
  }
}

export function fetchLinkTitle(url: string): Promise<string> {
  const normalizedUrl = normalizeExternalUrl(url)
  const key = titleCacheKey(normalizedUrl)

  if (!isTitleFetchable(normalizedUrl)) {
    return Promise.resolve('')
  }

  if (titleCache.has(key)) {
    return Promise.resolve(titleCache.get(key) ?? '')
  }

  const pending = titleInflight.get(key)

  if (pending) {
    return pending
  }

  const promise = fetchHtmlTitle(normalizedUrl)
    .then(value => usableTitle(value))
    .then(clean => {
      cacheTitle(key, clean)
      titleInflight.delete(key)
      titleSubs.get(key)?.forEach(sub => sub(clean))

      return clean
    })

  titleInflight.set(key, promise)

  return promise
}

export function useLinkTitle(url?: null | string): string {
  const normalizedUrl = useMemo(() => (url ? normalizeExternalUrl(url) : ''), [url])
  const key = useMemo(() => (normalizedUrl ? titleCacheKey(normalizedUrl) : ''), [normalizedUrl])
  const [title, setTitle] = useState(() => (key ? (titleCache.get(key) ?? '') : ''))

  useEffect(() => {
    setTitle(key ? (titleCache.get(key) ?? '') : '')

    if (!key || !isTitleFetchable(normalizedUrl)) {
      return
    }

    const subs = titleSubs.get(key) ?? new Set<(value: string) => void>()

    subs.add(setTitle)
    titleSubs.set(key, subs)
    void fetchLinkTitle(normalizedUrl)

    return () => {
      subs.delete(setTitle)

      if (!subs.size) {
        titleSubs.delete(key)
      }
    }
  }, [key, normalizedUrl])

  return title
}

export function __resetLinkTitleCache(): void {
  titleCache.clear()
  titleInflight.clear()
  titleSubs.clear()
}
