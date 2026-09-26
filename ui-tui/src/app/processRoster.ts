import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useMemo } from 'react'

import { type Locale, translate } from '../i18n/index.js'

import { $uiState } from './uiStore.js'

// Background `terminal(background=true)` processes owned by this session, as the
// gateway's `process.list` reports them. Session-local presentation only.

export interface ProcessEntry {
  session_id: string
  command?: string
  status?: string
  uptime_seconds?: number | null
  exit_code?: number | null
  exited_at?: number | null
  completion_reason?: string | null
  output_preview?: string
}

export interface ProcessRow {
  command: string
  /** Latest non-empty output line while running; the exit verdict once finished. */
  detail: string
  elapsedSeconds: number
  id: string
  /** Seconds since exit; 0 while running. */
  sinceExitSeconds: number
  status: 'done' | 'failed' | 'killed' | 'lost' | 'running'
}

/** A finished process stays on the dock long enough to read its exit line, then
 * leaves; the completion notification in the transcript is the durable record. */
export const PROCESS_RETAIN_SECONDS = 60

export const $processSnapshot = atom<{ sid: string | null; processes: ProcessEntry[] }>({ sid: null, processes: [] })

export function applyProcessSnapshot(sid: string | null, processes: ProcessEntry[] = []) {
  const previous = $processSnapshot.get()

  if (previous.sid !== sid || JSON.stringify(previous.processes) !== JSON.stringify(processes)) {
    $processSnapshot.set({ sid, processes })
  }
}

const REASON_STATUS: Record<string, ProcessRow['status']> = { failed_start: 'failed', killed: 'killed', lost: 'lost' }

export const processStatus = (entry: ProcessEntry): ProcessRow['status'] => {
  if (entry.status !== 'exited') {
    return 'running'
  }

  return REASON_STATUS[entry.completion_reason ?? ''] ?? (entry.exit_code ? 'failed' : 'done')
}

const lastOutputLine = (preview: string | undefined): string => {
  for (const line of (preview ?? '').split('\n').reverse()) {
    const text = line.replace(/\s+/g, ' ').trim()

    if (text) {
      return text
    }
  }

  return ''
}

export const processVerdict = (row: ProcessRow, exitCode: number | null | undefined, locale: Locale = 'en'): string => {
  if (row.status === 'running') {
    return row.detail
      ? translate(locale, 'process.last', { detail: row.detail })
      : translate(locale, 'process.starting')
  }

  const verdict =
    row.status === 'killed' || row.status === 'lost'
      ? translate(locale, `process.${row.status}`)
      : translate(locale, 'process.exit', { code: exitCode ?? '?' })

  return translate(locale, 'process.ago', { verdict, seconds: row.sinceExitSeconds })
}

/** Running processes first (longest running first), then recently exited ones
 * newest-exit first; exits older than the retention window are dropped. */
export const buildProcessRows = (
  processes: readonly ProcessEntry[],
  nowMs: number,
  locale: Locale = 'en'
): ProcessRow[] => {
  const nowS = nowMs / 1000
  const rows: ProcessRow[] = []

  for (const entry of processes) {
    const status = processStatus(entry)
    const exitedAt = status === 'running' ? 0 : (entry.exited_at ?? 0)
    const sinceExitSeconds = exitedAt ? Math.max(0, Math.floor(nowS - exitedAt)) : 0

    if (status !== 'running' && (!exitedAt || sinceExitSeconds > PROCESS_RETAIN_SECONDS)) {
      continue
    }

    const row: ProcessRow = {
      command: (entry.command ?? '').replace(/\s+/g, ' ').trim() || translate(locale, 'process.background'),
      detail: lastOutputLine(entry.output_preview),
      elapsedSeconds: Math.max(0, entry.uptime_seconds ?? 0) - sinceExitSeconds,
      id: entry.session_id,
      sinceExitSeconds,
      status
    }

    row.detail = processVerdict(row, entry.exit_code, locale)
    rows.push(row)
  }

  return rows.sort((a, b) =>
    (a.status === 'running') !== (b.status === 'running')
      ? a.status === 'running'
        ? -1
        : 1
      : a.status === 'running'
        ? b.elapsedSeconds - a.elapsedSeconds
        : a.sinceExitSeconds - b.sinceExitSeconds
  )
}

export function useProcessRows(nowMs: number): ProcessRow[] {
  const snapshot = useStore($processSnapshot)
  const { sid, locale } = useStore($uiState)

  return useMemo(
    () => buildProcessRows(snapshot.sid === sid ? snapshot.processes : [], nowMs, locale),
    [snapshot, sid, nowMs, locale]
  )
}
