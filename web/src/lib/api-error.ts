import { en } from '@/i18n/en'
import type { Translations } from '@/i18n/types'
/**
 * User-facing error shape for the dashboard REST layer.
 *
 * `fetchJSON` used to throw `new Error("<status>: <raw body>")`, and ~40 toast
 * call sites interpolated the Error object (`Error: ${e}`), so users saw
 * `Error: Error: 404: {"detail":"Server 'foo' not found"}`. This module turns
 * that into a plain sentence (`Server 'foo' not found`) while keeping the
 * status and raw body on the error object for diagnostics.
 */

/** The dashboard's own backend could not be reached at all (fetch rejected). */
export const API_UNREACHABLE_MESSAGE = en.common.apiError0

/** Status → plain sentence, used when the body carries no usable `detail`. */
const STATUS_COPY: Record<number, keyof Translations['common']> = {
  400: 'apiError400',
  401: 'apiError401',
  403: 'apiError403',
  404: 'apiError404',
  409: 'apiError409',
  413: 'apiError413',
  422: 'apiError422',
  429: 'apiError429',
  500: 'apiError500',
  502: 'apiError502',
  503: 'apiError503',
  504: 'apiError504'
}

export function humanizeStatus(status: number, copy: Translations['common'] = en.common): string {
  return STATUS_COPY[status]
    ? copy[STATUS_COPY[status]]
    : copy.apiErrorUnexpected.replace('{status}', () => String(status))
}

/** Pull a human sentence out of a FastAPI-style error body, or null. */
export function extractDetail(body: string): string | null {
  const text = body.trim()
  if (!text) return null
  if (text.startsWith('{') || text.startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(text)
      if (parsed && typeof parsed === 'object') {
        const record = parsed as Record<string, unknown>
        for (const key of ['detail', 'message', 'error']) {
          const value = record[key]
          if (typeof value === 'string' && value.trim()) return value.trim()
          // FastAPI validation errors: detail is a list of {msg, loc}.
          if (Array.isArray(value)) {
            const msgs = value
              .map(item => (item && typeof item === 'object' ? (item as { msg?: unknown }).msg : null))
              .filter((m): m is string => typeof m === 'string')
            if (msgs.length) return msgs.join('; ')
          } else if (value && typeof value === 'object') {
            // Structured detail ({error: "<code>", message: "<sentence>"}): the code is for
            // programs, the sentence is what the operator needs to act on.
            const message = (value as { message?: unknown }).message
            if (typeof message === 'string' && message.trim()) return message.trim()
          }
        }
      }
      return null
    } catch {
      return null
    }
  }
  if (text.startsWith('<')) return null // HTML error page from a proxy
  return text.length <= 300 ? text : null
}

export class ApiError extends Error {
  /** HTTP status; 0 when the request never reached the server. */
  readonly status: number
  /** Raw response body (or the transport error text) for a Copy-details action. */
  readonly body: string
  /** Request URL (path only), for diagnostics. */
  readonly url: string
  /** A known local error is translated only at the presentation boundary. */
  presentationStatus?: number

  constructor(message: string, init: { status: number; body: string; url: string }) {
    super(message)
    this.name = 'ApiError'
    this.status = init.status
    this.body = init.body
    this.url = init.url
  }

  /** Multi-line technical detail for a Copy-details action or console. */
  get details(): string {
    const head = this.status ? `HTTP ${this.status} ${this.url}` : `network failure ${this.url}`
    return this.body ? `${head}\n${this.body}` : head
  }
}

export function apiErrorFromResponse(status: number, body: string, url: string): ApiError {
  const detail = extractDetail(body)
  const error = new ApiError(detail ?? humanizeStatus(status), { status, body, url })
  if (!detail) error.presentationStatus = status
  return error
}

export function apiErrorFromNetworkFailure(cause: unknown, url: string): ApiError {
  const body = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause)
  const error = new ApiError(API_UNREACHABLE_MESSAGE, { status: 0, body, url })
  error.presentationStatus = 0
  return error
}

/**
 * The one way to turn a caught `unknown` into toast/inline text. Never yields
 * a `Error: Error:` double prefix because it reads `.message`, not `String(e)`.
 */
export function errorMessage(err: unknown, copy: Translations['common'] = en.common): string {
  if (err instanceof ApiError && err.presentationStatus !== undefined) {
    return err.presentationStatus === 0 ? copy.apiError0 : humanizeStatus(err.presentationStatus, copy)
  }
  if (err instanceof Error) return err.message || err.name
  if (typeof err === 'string') return err
  return String(err)
}
