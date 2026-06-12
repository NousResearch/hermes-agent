/**
 * SessionPicker — the tabbed resume picker (design doc
 * docs/plans/opentui-resume-picker.md §A; supersedes SessionSwitcher).
 *
 * Layout per the §A mock: search `<input>` (focused throughout — the model
 * picker's input+global-useKeyboard discipline, view/overlays/picker.tsx),
 * the reusable TabChips strip (Recent/Cron/Gateways/All), a windowed row list
 * driven programmatically (↑↓/Enter from the global key handler with
 * preventDefault, so the input never also applies them), a Space-toggled
 * preview pane fed by `session.peek`, Ctrl+R inline rename via
 * `session.title`, and the spec footer. Esc/Ctrl+C close through BOTH the
 * keymap layer and the global handler (picker pattern) — funneled through a
 * once-per-press guard so one Esc can't cancel the rename AND close.
 *
 * Data flow: each tab queries `session.list` with its `sources` allow-list
 * (logic/sessionPicker.ts `listParamsFor`); the search filters CLIENT-side
 * within the fetched rows (fuzzy over title/preview/cwd/id). Pagination is a
 * selectable "load more" row (offset = rows fetched so far) — shown until a
 * short page signals exhaustion; the gateway's `truncated` flag renders an
 * honest notice instead of pretending the page is final.
 *
 * Peek debounce: selection changes while the preview is open schedule the
 * peek after PEEK_DEBOUNCE_MS (holding ↓ scrolls without a peek per row);
 * the pending timer is cleared on every change and a monotonic seq drops
 * stale responses, so exactly one peek lands per SETTLED highlight.
 */
import type { BoxRenderable, InputRenderable } from '@opentui/core'
import { useKeyboard } from '@opentui/solid'
import { Option } from 'effect'
import { createEffect, createMemo, createSignal, For, on, onCleanup, Show } from 'solid-js'

import { decodeSessionPeek, type SessionPeekDecoded } from '../../boundary/schema/SessionPeek.ts'
import { visibleRows, type PickerRow } from '../../logic/fuzzy.ts'
import {
  filterSessions,
  listParamsFor,
  orderRowsForCwd,
  mapSessionRows,
  relativeTime,
  routeSessionPickerKey,
  rowMeta,
  SESSION_TABS,
  tailTruncate,
  type SessionRow,
  type SessionTabId
} from '../../logic/sessionPicker.ts'
import { useCloseLayer } from '../keymap.tsx'
import { useTheme } from '../theme.tsx'
import { TabChips } from './picker.tsx'

/** Gateway calls the picker needs (wired from the entry; fakeable in tests). */
export interface SessionPickerOps {
  /** Raw `session.list` result for the given params. */
  readonly list: (params: Record<string, unknown>) => Promise<unknown>
  /** Raw `session.peek` result for a session id. */
  readonly peek: (sessionId: string) => Promise<unknown>
  /** Rename via `session.title` (rejects when the gateway can't — surfaced). */
  readonly rename: (sessionId: string, title: string) => Promise<void>
}

/** Page size per `session.list` fetch (a "load more" row pulls the next). */
const PAGE_SIZE = 100
/** Debounce for preview refresh while the selection changes (ms). */
export const PEEK_DEBOUNCE_MS = 120
/** Max session rows visible at once (each row renders 2 lines). */
const MAX_ROWS = 8
/** Max visible rows while the preview pane is open (it needs the space). */
const MAX_ROWS_PREVIEW = 4
/** One-line cap for preview message excerpts. */
const PEEK_EXCERPT = 110

/** First line of a message, capped — preview rows must stay one line each. */
function excerpt(text: string): string {
  const line = text.split('\n', 1)[0] ?? ''
  return line.length > PEEK_EXCERPT ? `${line.slice(0, PEEK_EXCERPT - 1)}…` : line
}

export function SessionPicker(props: {
  ops: SessionPickerOps
  onResume: (sessionId: string) => void
  onClose: () => void
  initialTab?: SessionTabId
  /** The TUI's working directory — sessions started here group first while
   *  browsing (no effect during search). */
  currentCwd?: () => string | undefined
  /** Test seam: override the peek debounce (default PEEK_DEBOUNCE_MS). */
  peekDebounceMs?: number
}) {
  const theme = useTheme()
  let rootRef: BoxRenderable | undefined
  let searchRef: InputRenderable | undefined
  let renameRef: InputRenderable | undefined

  // ── core state ──────────────────────────────────────────────────────────
  const [tab, setTab] = createSignal<SessionTabId>(props.initialTab ?? 'recent')
  const [query, setQuery] = createSignal('')
  const [rows, setRows] = createSignal<SessionRow[]>([])
  const [truncated, setTruncated] = createSignal(false)
  const [exhausted, setExhausted] = createSignal(false)
  const [loading, setLoading] = createSignal(false)
  const [listError, setListError] = createSignal(false)
  const [sel, setSel] = createSignal(0)
  // transient note line (rename failures and the like)
  const [note, setNote] = createSignal<string | undefined>()

  // ── session.list fetching (per tab; "load more" appends the next page) ──
  // A monotonic seq drops responses from a superseded tab/page request.
  let listSeq = 0
  const fetchPage = (offset: number) => {
    const seq = ++listSeq
    setLoading(true)
    setListError(false)
    props.ops
      .list(listParamsFor(tab(), offset, PAGE_SIZE))
      .then(raw => {
        if (seq !== listSeq) return
        const page = mapSessionRows(raw)
        setRows(prev => (offset === 0 ? page.rows : [...prev, ...page.rows]))
        setTruncated(page.truncated)
        setExhausted(page.rows.length < PAGE_SIZE)
      })
      .catch(() => {
        if (seq === listSeq) setListError(true)
      })
      .finally(() => {
        if (seq === listSeq) setLoading(false)
      })
  }
  // Tab activation (incl. the initial one) resets and re-queries with that
  // tab's `sources` — search-within-tab, so the query survives the switch.
  createEffect(
    on(tab, () => {
      setRows([])
      setTruncated(false)
      setExhausted(false)
      setSel(0)
      fetchPage(0)
    })
  )

  // Display order: this-directory sessions first while browsing (design ask
  // 2026-06-12); search keeps pure fuzzy relevance. One flat list, so the
  // selection/windowing math below is order-agnostic.
  const ordered = createMemo(() => orderRowsForCwd(filterSessions(query(), rows()), props.currentCwd?.(), query()))
  const filtered = createMemo(() => ordered().rows)
  const hereCount = createMemo(() => ordered().hereCount)
  createEffect(on(query, () => setSel(0), { defer: true }))

  // "load more" pseudo-row: selectable index === filtered().length. Offered
  // until a short page proves the tab exhausted (query or not — more matches
  // may live in unfetched pages).
  const hasMore = createMemo(() => !exhausted() && rows().length > 0)
  const selectableCount = createMemo(() => filtered().length + (hasMore() ? 1 : 0))
  /** The highlighted session row (undefined on the load-more row / empty list). */
  const highlighted = createMemo(() => filtered()[sel()])

  const move = (dir: 1 | -1) => {
    const count = selectableCount()
    if (count) setSel(s => (s + dir + count) % count)
  }

  const cycleTab = (dir: 1 | -1) => {
    const at = SESSION_TABS.findIndex(t => t.id === tab())
    const next = SESSION_TABS[(at + dir + SESSION_TABS.length) % SESSION_TABS.length]
    if (next) setTab(next.id)
  }

  // ── preview (Space → session.peek; refreshed on highlight change) ───────
  const [previewOpen, setPreviewOpen] = createSignal(false)
  const [peekInfo, setPeekInfo] = createSignal<SessionPeekDecoded | undefined>()
  const [peekState, setPeekState] = createSignal<'loading' | 'ready' | 'error'>('loading')
  let peekSeq = 0
  let peekTimer: ReturnType<typeof setTimeout> | undefined
  const cancelPendingPeek = () => {
    if (peekTimer) clearTimeout(peekTimer)
    peekTimer = undefined
  }
  onCleanup(cancelPendingPeek)
  const issuePeek = (id: string) => {
    const seq = ++peekSeq
    setPeekState('loading')
    props.ops
      .peek(id)
      .then(raw => {
        if (seq !== peekSeq) return // stale — a newer row was highlighted
        const decoded = decodeSessionPeek(raw)
        if (Option.isNone(decoded)) {
          setPeekInfo(undefined)
          setPeekState('error')
        } else {
          setPeekInfo(decoded.value)
          setPeekState('ready')
        }
      })
      .catch(() => {
        if (seq === peekSeq) {
          setPeekInfo(undefined)
          setPeekState('error')
        }
      })
  }
  // One reactive key drives ALL peek fetches: the highlighted id while the
  // pane is open (memo dedupes by value). First open fires immediately;
  // subsequent highlight moves debounce, cancelling the pending timer —
  // holding ↓ settles into exactly one peek.
  const peekKey = createMemo(() => (previewOpen() ? (highlighted()?.id ?? undefined) : undefined))
  createEffect(
    on(peekKey, (id, prev) => {
      cancelPendingPeek()
      if (!id) return
      if (prev === undefined) issuePeek(id)
      else peekTimer = setTimeout(() => issuePeek(id), props.peekDebounceMs ?? PEEK_DEBOUNCE_MS)
    })
  )
  const togglePreview = () => {
    if (previewOpen()) {
      cancelPendingPeek()
      peekSeq++ // in-flight responses are now stale
      setPreviewOpen(false)
      setPeekInfo(undefined)
    } else if (highlighted()) {
      setPreviewOpen(true)
    }
  }

  // ── inline rename (Ctrl+R → session.title) ──────────────────────────────
  const [renaming, setRenaming] = createSignal(false)
  const startRename = () => {
    if (highlighted()) setRenaming(true)
  }
  const endRename = () => {
    setRenaming(false)
    searchRef?.focus()
  }
  const commitRename = () => {
    const row = highlighted()
    const title = (renameRef?.value ?? '').trim()
    endRename()
    if (!row || !title || title === row.title) return
    props.ops
      .rename(row.id, title)
      .then(() => {
        setRows(prev => prev.map(r => (r.id === row.id ? { ...r, title } : r)))
        setNote(`renamed → ${title}`)
      })
      .catch(error => {
        // session.title only reaches LIVE gateway sessions — old rows reject;
        // surface it instead of silently dropping the edit.
        setNote(`rename failed: ${error instanceof Error ? error.message : 'gateway rejected'}`)
      })
  }

  // One Esc/Ctrl+C press can reach us TWICE (the keymap close layer + the
  // global handler — the picker runs both so close works even when focus
  // never landed). The guard makes the press act once, so cancelling a rename
  // never ALSO closes the picker.
  let lastEscape = 0
  const oncePerPress = (fn: () => void) => {
    const now = Date.now()
    if (now - lastEscape < 50) return
    lastEscape = now
    fn()
  }
  const closeRequest = () => oncePerPress(() => (renaming() ? endRename() : props.onClose()))
  useCloseLayer(
    () => rootRef,
    () => closeRequest()
  )

  const activate = () => {
    const row = highlighted()
    if (row) return props.onResume(row.id)
    // the load-more row: next offset = rows fetched so far (server ordering)
    if (hasMore() && sel() === filtered().length && !loading()) fetchPage(rows().length)
  }

  useKeyboard(key => {
    const action = routeSessionPickerKey(
      key.name,
      { ctrl: key.ctrl, shift: key.shift },
      { queryEmpty: !query(), renaming: renaming() }
    )
    if (action.kind === 'pass') return
    // every routed chord is consumed BEFORE the focused input sees it (Space
    // would insert ' ', Enter would submit, Ctrl+R/↑↓ are native edits).
    key.preventDefault()
    switch (action.kind) {
      case 'close':
        return closeRequest()
      case 'cancel-rename':
        return closeRequest()
      case 'commit-rename':
        return commitRename()
      case 'resume':
        return activate()
      case 'move':
        return move(action.dir)
      case 'cycle-tab':
        return cycleTab(action.dir)
      case 'preview':
        return togglePreview()
      case 'rename':
        return startRename()
    }
  })

  // ── render ──────────────────────────────────────────────────────────────
  const maxRows = () => (previewOpen() ? MAX_ROWS_PREVIEW : MAX_ROWS)
  const win = createMemo(() => {
    const items: PickerRow<SessionRow | undefined>[] = filtered().map((item, index) => ({ index, item, kind: 'item' }))
    if (hasMore()) items.push({ index: filtered().length, item: undefined, kind: 'item' })
    return visibleRows(items, sel(), maxRows())
  })
  const nowMs = Date.now() // stamped once at open — the overlay is short-lived
  const title = () => `⟲ Resume session (${filtered().length} of ${rows().length}${exhausted() ? '' : '+'})`
  const peekMeta = (p: SessionPeekDecoded): string => {
    const s = p.session
    const parts: string[] = []
    if (s?.model) parts.push(s.model)
    if (s?.cwd) parts.push(tailTruncate(s.cwd, 40))
    if (typeof s?.cost_usd === 'number') parts.push(`$${s.cost_usd.toFixed(2)}`)
    parts.push(`${p.total_messages ?? s?.message_count ?? 0} msgs`)
    if (s?.last_active) parts.push(relativeTime(s.last_active, nowMs))
    return parts.join(' · ')
  }

  return (
    <box
      ref={el => (rootRef = el)}
      style={{ borderColor: theme().color.border, flexDirection: 'column', flexShrink: 0, marginTop: 1, padding: 1 }}
      border
    >
      {/* title + search line (input keeps focus the whole time) */}
      <box style={{ flexDirection: 'row' }}>
        <text fg={theme().color.accent}>
          <b>{title()}</b>
        </text>
        <text fg={theme().color.label}>{'  '}</text>
        <text fg={theme().color.prompt}>{'⌕ '}</text>
        <input
          ref={el => (searchRef = el)}
          focused
          onInput={setQuery}
          onMouseDown={() => searchRef?.focus()}
          placeholder="search title · preview · cwd · id"
          placeholderColor={theme().color.muted}
          textColor={theme().color.text}
          cursorColor={theme().color.accent}
          backgroundColor="transparent"
          focusedBackgroundColor="transparent"
          style={{ flexGrow: 1, minWidth: 0 }}
        />
        <Show when={loading()}>
          <text fg={theme().color.muted}>loading…</text>
        </Show>
      </box>
      <TabChips labels={SESSION_TABS.map(t => t.label)} active={SESSION_TABS.findIndex(t => t.id === tab())} />
      {/* inline rename line (Ctrl+R) — its input takes focus while open */}
      <Show when={renaming()}>
        <box style={{ flexDirection: 'row' }}>
          <text fg={theme().color.accent}>rename: </text>
          <input
            ref={el => {
              renameRef = el
              el.value = highlighted()?.title ?? ''
              el.focus()
            }}
            textColor={theme().color.text}
            cursorColor={theme().color.accent}
            backgroundColor="transparent"
            focusedBackgroundColor="transparent"
            style={{ flexGrow: 1, minWidth: 0 }}
          />
        </box>
      </Show>
      <Show when={win().above > 0}>
        <text fg={theme().color.muted}>{`  ↑ ${win().above} more`}</text>
      </Show>
      <For each={win().rows}>
        {row =>
          row.kind === 'item' && row.item ? (
            <box style={{ flexDirection: 'column' }} onMouseDown={() => setSel(row.index)}>
              <Show when={hereCount() > 0 && row.index === 0}>
                <text fg={theme().color.muted}>{`  ▾ this directory (${hereCount()})`}</text>
              </Show>
              <Show when={hereCount() > 0 && row.index === hereCount()}>
                <text fg={theme().color.muted}>{'  ▾ other directories'}</text>
              </Show>
              <text bg={row.index === sel() ? theme().color.selectionBg : 'transparent'}>
                <span style={{ fg: row.index === sel() ? theme().color.text : theme().color.muted }}>
                  {row.index === sel() ? '❯ ' : '  '}
                </span>
                <span style={{ fg: theme().color.text }}>
                  {row.item.title || excerpt(row.item.preview) || row.item.id}
                </span>
              </text>
              <text fg={theme().color.muted}>{`  ${rowMeta(row.item, nowMs)}`}</text>
            </box>
          ) : (
            // the load-more pseudo-row (kind 'item' with no session behind it)
            <text bg={row.kind === 'item' && row.index === sel() ? theme().color.selectionBg : 'transparent'}>
              <span style={{ fg: theme().color.muted }}>
                {row.kind === 'item' && row.index === sel() ? '❯ ' : '  '}
              </span>
              <span style={{ fg: theme().color.muted }}>
                {loading() ? 'loading…' : `↓ load more (${rows().length} loaded)`}
              </span>
            </text>
          )
        }
      </For>
      <Show when={!loading() && filtered().length === 0}>
        <text fg={theme().color.muted}>
          {listError() ? ' (session list unavailable)' : query() ? ' (no matches)' : ' (no sessions on this tab)'}
        </text>
      </Show>
      <Show when={truncated()}>
        <text fg={theme().color.warn}> results truncated — narrow the search</text>
      </Show>
      {/* preview pane (Space) — head/tail excerpts from session.peek */}
      <Show when={previewOpen()}>
        <box
          style={{ borderColor: theme().color.border, flexDirection: 'column', flexShrink: 0 }}
          border
          title="preview (Space)"
        >
          <Show when={peekState() === 'loading'}>
            <text fg={theme().color.muted}>loading preview…</text>
          </Show>
          <Show when={peekState() === 'error'}>
            <text fg={theme().color.muted}>preview unavailable</text>
          </Show>
          <Show when={peekState() === 'ready' && peekInfo()}>
            {p => (
              <>
                <text fg={theme().color.label}>{peekMeta(p())}</text>
                <For each={p().head ?? []}>
                  {m => (
                    <text>
                      <span style={{ fg: theme().color.accent }}>{`${m.role ?? '?'} › `}</span>
                      <span style={{ fg: theme().color.text }}>{excerpt(m.content ?? '')}</span>
                    </text>
                  )}
                </For>
                <Show when={(p().total_messages ?? 0) > (p().head?.length ?? 0) + (p().tail?.length ?? 0)}>
                  <text fg={theme().color.muted}>{`  ⋯ ${p().total_messages} messages ⋯`}</text>
                </Show>
                <For each={p().tail ?? []}>
                  {m => (
                    <text>
                      <span style={{ fg: theme().color.accent }}>{`${m.role ?? '?'} › `}</span>
                      <span style={{ fg: theme().color.text }}>{excerpt(m.content ?? '')}</span>
                    </text>
                  )}
                </For>
              </>
            )}
          </Show>
        </box>
      </Show>
      <Show when={note()}>
        <text fg={theme().color.muted}>{note()}</text>
      </Show>
      <text fg={theme().color.muted}>
        {renaming()
          ? 'Enter save · Esc cancel rename'
          : '↑↓ select · Enter resume · Tab scope · Space preview · Ctrl+R rename · Esc cancel'}
      </text>
    </box>
  )
}
