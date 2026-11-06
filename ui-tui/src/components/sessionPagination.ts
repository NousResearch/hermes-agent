import type { SessionListRow } from '@hermes/shared/gateway-events'

import type { SessionActiveItem } from '../gatewayTypes.js'
import { messages } from '../i18n/runtime.js'
import { asRpcResult } from '../lib/rpc.js'

export const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER
export const DEFAULT_PAGE_LIMIT = 200

export interface SessionPageMetadata {
  offset: number
  limit: number
  has_more: boolean
}

export type ValidatedPageResult =
  | { kind: 'legacy-unpaged'; sessions: SessionListRow[] }
  | { kind: 'malformed'; error: string }
  | { kind: 'valid'; meta: SessionPageMetadata; sessions: SessionListRow[] }

export type SwitcherWindowedRow =
  | { item: SessionActiveItem; kind: 'live' }
  | { item: SessionListRow; kind: 'history' }
  | { kind: 'pagination-newer'; targetOffset: number }
  | { kind: 'pagination-older'; targetOffset: number }

function isValidOffset(offset: unknown, expectedOffset: number): offset is number {
  if (typeof offset !== 'number' || !Number.isInteger(offset)) {
    return false
  }

  return offset >= 0 && offset <= MAX_SAFE_INTEGER && offset === expectedOffset
}

function isValidLimit(limit: unknown): limit is number {
  if (typeof limit !== 'number' || !Number.isInteger(limit)) {
    return false
  }

  return limit >= 1 && limit <= DEFAULT_PAGE_LIMIT
}

function isPagePayloadValid(sessionsLen: number, offset: number, limit: number, hasMore: boolean): boolean {
  if (sessionsLen > limit) {
    return false
  }

  if (sessionsLen === 0 && hasMore) {
    return false
  }

  if (hasMore && offset + limit > MAX_SAFE_INTEGER) {
    return false
  }

  return true
}

function isValidSessionRow(value: unknown): value is SessionListRow {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const row = value as Record<string, unknown>

  return (
    typeof row.id === 'string' &&
    ['title', 'preview'].every(key => row[key] == null || typeof row[key] === 'string') &&
    ['started_at', 'message_count'].every(
      key => row[key] === undefined || (typeof row[key] === 'number' && Number.isFinite(row[key]))
    )
  )
}

/**
 * Validates rows and metadata returned from `session.list`.
 *
 * Rules:
 * 1. Supports JSON-RPC result envelopes (via asRpcResult) as well as direct objects.
 * 2. Require an array of objects with string IDs and correctly typed optional display fields.
 * 3. If offset, limit, and has_more are all omitted (undefined):
 *    - If isInitial is true -> legacy-unpaged (older gateway).
 *    - If !isInitial (including later Newer 0 navigations) -> malformed (never downgrade).
 * 4. If metadata is present, strictly check:
 *    - offset: integer, >= 0, <= MAX_SAFE_INTEGER, must equal expectedOffset.
 *    - limit: integer, >= 1, <= 200.
 *    - has_more: boolean.
 *    - sessions.length <= limit.
 *    - !(sessions.length === 0 && has_more === true).
 *    - Unsafe next offset (offset + limit > MAX_SAFE_INTEGER) is validated ONLY when has_more is true.
 */
export function validateSessionPageResponse(
  raw: unknown,
  expectedOffset: number,
  isInitial = false
): ValidatedPageResult {
  const unwrapped = asRpcResult(raw) ?? (raw && typeof raw === 'object' ? raw : null)

  if (!unwrapped || typeof unwrapped !== 'object') {
    return { error: messages().pickers.common.invalidResponse('session.list'), kind: 'malformed' }
  }

  const rawObj = unwrapped as Record<string, unknown>
  const innerResult = rawObj.result
  const obj = (
    innerResult && typeof innerResult === 'object' && !Array.isArray(innerResult)
      ? innerResult
      : rawObj
  ) as Record<string, unknown>

  const sessions = obj.sessions

  if (!Array.isArray(sessions) || !sessions.every(isValidSessionRow)) {
    return { error: messages().pickers.common.invalidResponse('session.list'), kind: 'malformed' }
  }

  const hasMeta = obj.offset !== undefined || obj.limit !== undefined || obj.has_more !== undefined

  if (!hasMeta) {
    if (isInitial && sessions.length <= DEFAULT_PAGE_LIMIT) {
      return { kind: 'legacy-unpaged', sessions: sessions as SessionListRow[] }
    }

    return { error: messages().pickers.session.errors.couldNotLoadResumable, kind: 'malformed' }
  }

  const { offset, limit, has_more: hasMore } = obj

  if (!isValidOffset(offset, expectedOffset) || !isValidLimit(limit) || typeof hasMore !== 'boolean') {
    return { error: messages().pickers.common.invalidResponse('session.list'), kind: 'malformed' }
  }

  if (!isPagePayloadValid(sessions.length, offset, limit, hasMore)) {
    return { error: messages().pickers.common.invalidResponse('session.list'), kind: 'malformed' }
  }

  return {
    kind: 'valid',
    meta: {
      has_more: hasMore,
      limit,
      offset
    },
    sessions: sessions as SessionListRow[]
  }
}

export function buildSwitcherWindowedRows(
  liveItems: readonly SessionActiveItem[],
  historyItems: readonly SessionListRow[],
  offset: number,
  limit: number,
  hasMore: boolean
): SwitcherWindowedRow[] {
  const rows: SwitcherWindowedRow[] = []

  for (const item of liveItems) {
    rows.push({ item, kind: 'live' })
  }

  if (offset > 0) {
    rows.push({
      kind: 'pagination-newer',
      targetOffset: Math.max(0, offset - limit)
    })
  }

  for (const item of historyItems) {
    rows.push({ item, kind: 'history' })
  }

  if (hasMore && offset + limit <= MAX_SAFE_INTEGER) {
    rows.push({
      kind: 'pagination-older',
      targetOffset: offset + limit
    })
  }

  return rows
}

export function resolvePageChangeSelection(
  prevKind: string,
  prevId: string | null,
  nextLive: readonly SessionActiveItem[],
  nextHistory: readonly SessionListRow[],
  nextWindowedRows: readonly SwitcherWindowedRow[]
): number {
  if (prevKind === 'new') {
    return 0
  }

  if (prevKind === 'live' && prevId) {
    const rowIdx = nextWindowedRows.findIndex(r => r.kind === 'live' && r.item.id === prevId)
    if (rowIdx >= 0) {
      return rowIdx + 1
    }

    return 0
  }

  if (prevKind === 'history' && prevId) {
    const histIdx = nextHistory.findIndex(h => h.id === prevId)
    if (histIdx >= 0) {
      const rowIdx = nextWindowedRows.findIndex(r => r.kind === 'history' && r.item.id === prevId)
      if (rowIdx >= 0) {
        return rowIdx + 1
      }
    }
  }

  // Documented fallback:
  // "If selected history identity is absent on a successful explicit page change,
  // select first history row (or new row when none); no accidental auto-resume."
  const firstHistIdx = nextWindowedRows.findIndex(r => r.kind === 'history')
  if (firstHistIdx >= 0) {
    return firstHistIdx + 1
  }

  return 0
}

export function resolveLivePollSelection(
  currentSel: number,
  prevKind: string,
  prevId: string | null,
  nextWindowedRows: readonly SwitcherWindowedRow[]
): number {
  if (currentSel <= 0 || prevKind === 'new') {
    return 0
  }

  if (prevKind === 'live' && prevId) {
    const rowIdx = nextWindowedRows.findIndex(r => r.kind === 'live' && r.item.id === prevId)
    if (rowIdx >= 0) {
      return rowIdx + 1
    }
  }

  if (prevKind === 'history' && prevId) {
    const rowIdx = nextWindowedRows.findIndex(r => r.kind === 'history' && r.item.id === prevId)
    if (rowIdx >= 0) {
      return rowIdx + 1
    }
  }

  if (prevKind === 'pagination-newer') {
    const rowIdx = nextWindowedRows.findIndex(r => r.kind === 'pagination-newer')
    if (rowIdx >= 0) {
      return rowIdx + 1
    }
  }

  if (prevKind === 'pagination-older') {
    const rowIdx = nextWindowedRows.findIndex(r => r.kind === 'pagination-older')
    if (rowIdx >= 0) {
      return rowIdx + 1
    }
  }

  return Math.max(0, Math.min(currentSel, nextWindowedRows.length))
}
