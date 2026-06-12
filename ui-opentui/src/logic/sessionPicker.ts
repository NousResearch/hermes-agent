/**
 * sessionPicker.ts — pure logic for the tabbed resume picker (design doc
 * docs/plans/opentui-resume-picker.md §A/§B item 5; supersedes the flat
 * SessionSwitcher). Everything here is view-free and vitest-covered:
 *
 *   - tab definitions + source→tab classification (Recent = interactive
 *     cli/tui/acp + unknown/custom; Cron = cron; Gateways = the known platform
 *     sources; All = everything minus the deny-listed `tool`),
 *   - the `session.list` params each tab queries with (`sources` allow-list —
 *     note the one honest gap: unknown/custom sources CLASSIFY as Recent but
 *     can't be expressed in an allow-list, so they surface under All),
 *   - the client-side search filter chain over title/preview/cwd/id (reuses
 *     fuzzy.ts — same scorer as the model picker),
 *   - the key-routing decision table (pattern: completionMenu's routeMenuKey),
 *   - the relative-time formatter + row-meta composer (time · source · N msgs
 *     · tail-truncated cwd),
 *   - `/sessions <tab>` arg parsing and the `/resume <id|name>` resolver.
 */
import { fuzzyFilter, type FuzzyField } from './fuzzy.ts'

// ── tabs + classification ─────────────────────────────────────────────────

export type SessionTabId = 'recent' | 'cron' | 'gateways' | 'all'

/** Tab strip order + labels (design doc §A — Recent is the default). */
export const SESSION_TABS: ReadonlyArray<{ id: SessionTabId; label: string }> = [
  { id: 'recent', label: 'Recent' },
  { id: 'cron', label: 'Cron' },
  { id: 'gateways', label: 'Gateways' },
  { id: 'all', label: 'All' }
]

/** Interactive sources — the Recent tab's allow-list. */
export const INTERACTIVE_SOURCES: readonly string[] = ['cli', 'tui', 'acp']

/** Known platform/gateway sources (the Gateways tab's allow-list). The gateway
 *  itself deny-lists only `tool`, so this list is the picker's working set of
 *  "messaging platform" tags; new platforms join here (or show under All). */
export const PLATFORM_SOURCES: readonly string[] = [
  'telegram',
  'discord',
  'slack',
  'whatsapp',
  'signal',
  'imessage',
  'matrix',
  'teams',
  'email',
  'webhook',
  'x',
  'twitter',
  'mastodon',
  'irc',
  'mattermost'
]

/** Classify a session `source` tag into its home tab (`tool` = deny-listed —
 *  never shown). Unknown/custom sources (incl. empty) default to Recent per
 *  the design table: they're assumed interactive `HERMES_SESSION_SOURCE`s. */
export function classifySource(source: string | undefined): 'recent' | 'cron' | 'gateways' | 'tool' {
  const s = (source ?? '').trim().toLowerCase()
  if (s === 'tool') return 'tool'
  if (s === 'cron') return 'cron'
  if (PLATFORM_SOURCES.includes(s)) return 'gateways'
  return 'recent'
}

/** Whether a row with this source belongs on the given tab. */
export function tabAccepts(tab: SessionTabId, source: string | undefined): boolean {
  const cls = classifySource(source)
  if (cls === 'tool') return false
  return tab === 'all' || cls === tab
}

/**
 * The `session.list` params a tab queries with. Cron/Gateways push an exact
 * `sources` allow-list to the gateway; All omits it (the gateway deny-lists
 * `tool` itself). Recent sends the interactive allow-list — the one honest gap
 * vs `classifySource` (unknown/custom sources can't be allow-listed, so they
 * appear under All only); fetching everything and filtering client-side would
 * make Recent unusable in cron-heavy DBs (1500+ cron rows drown the page).
 */
export function listParamsFor(tab: SessionTabId, offset: number, limit: number): Record<string, unknown> {
  const base: Record<string, unknown> = { limit, offset }
  if (tab === 'recent') return { ...base, sources: [...INTERACTIVE_SOURCES] }
  if (tab === 'cron') return { ...base, sources: ['cron'] }
  if (tab === 'gateways') return { ...base, sources: [...PLATFORM_SOURCES] }
  return base
}

// ── session.list row mapping ──────────────────────────────────────────────

/** One picker row — the widened `session.list` projection (gateway 529d8084b). */
export interface SessionRow {
  id: string
  title: string
  preview: string
  source: string
  messageCount: number
  startedAt: number
  lastActive: number
  endedAt?: number
  model?: string
  cwd?: string
}

function readStr(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== 'object') return undefined
  const v = (value as { [k: string]: unknown })[key]
  return typeof v === 'string' ? v : undefined
}

function readNum(value: unknown, key: string): number {
  if (!value || typeof value !== 'object') return 0
  const v = (value as { [k: string]: unknown })[key]
  return typeof v === 'number' ? v : 0
}

/** Map a widened `session.list` result into rows + the honesty flag. */
export function mapSessionRows(result: unknown): { rows: SessionRow[]; truncated: boolean } {
  if (!result || typeof result !== 'object') return { rows: [], truncated: false }
  const sessions = (result as { sessions?: unknown }).sessions
  const truncated = (result as { truncated?: unknown }).truncated === true
  if (!Array.isArray(sessions)) return { rows: [], truncated }
  const rows: SessionRow[] = []
  for (const s of sessions) {
    const id = readStr(s, 'id')
    if (!id) continue
    const row: SessionRow = {
      id,
      lastActive: readNum(s, 'last_active') || readNum(s, 'started_at'),
      messageCount: readNum(s, 'message_count'),
      preview: readStr(s, 'preview') ?? '',
      source: readStr(s, 'source') ?? '',
      startedAt: readNum(s, 'started_at'),
      title: readStr(s, 'title') ?? ''
    }
    const endedAt = readNum(s, 'ended_at')
    if (endedAt) row.endedAt = endedAt
    const model = readStr(s, 'model')
    if (model) row.model = model
    const cwd = readStr(s, 'cwd')
    if (cwd) row.cwd = cwd
    rows.push(row)
  }
  return { rows, truncated }
}

// ── search filter chain (client-side, within the active tab) ──────────────

/** Fuzzy haystacks of a row: title ×2 (primary), preview, cwd, id. */
export function sessionFields(row: SessionRow): FuzzyField[] {
  const fields: FuzzyField[] = []
  if (row.title) fields.push({ text: row.title, weight: 2 })
  if (row.preview) fields.push({ text: row.preview })
  if (row.cwd) fields.push({ text: row.cwd })
  fields.push({ text: row.id })
  return fields
}

/** Filter + rank rows by the search query (empty → all rows, fetch order). */
export function filterSessions(query: string, rows: readonly SessionRow[]): SessionRow[] {
  return fuzzyFilter(query, rows, sessionFields)
}

// ── this-directory grouping ───────────────────────────────────────────────

/** Path equality for cwd grouping: trim + drop trailing slashes. Pure string
 *  work (no fs) — rows carry the gateway's already-absolute paths. */
export function normalizeCwd(path: string | undefined): string {
  return (path ?? '').trim().replace(/\/+$/, '')
}

/** Display order with sessions started in the CURRENT directory first.
 *
 * Browse mode only: while a search query is active the fuzzy score owns the
 * order (relevance beats locality), so `hereCount` is 0 and rows pass through.
 * Stable within both groups (each keeps the gateway's recency order). The
 * view renders section captions off `hereCount`; selection math is untouched
 * because this just reorders the one flat list.
 */
export function orderRowsForCwd(
  rows: SessionRow[],
  currentCwd: string | undefined,
  query: string
): { rows: SessionRow[]; hereCount: number } {
  const here = normalizeCwd(currentCwd)
  if (!here || query.trim()) return { hereCount: 0, rows }
  const local: SessionRow[] = []
  const elsewhere: SessionRow[] = []
  for (const row of rows) (normalizeCwd(row.cwd) === here ? local : elsewhere).push(row)
  if (!local.length) return { hereCount: 0, rows }
  return { hereCount: local.length, rows: [...local, ...elsewhere] }
}

// ── key routing (pattern: completionMenu.ts routeMenuKey) ────────────────

export interface SessionPickerKeyContext {
  /** Whether the inline Ctrl+R rename is active (it owns Enter/Esc). */
  renaming: boolean
  /** Whether the search query is empty (←/→ only cycle tabs when it is). */
  queryEmpty: boolean
}

export type SessionPickerAction =
  | { kind: 'close' }
  | { kind: 'resume' }
  | { kind: 'move'; dir: 1 | -1 }
  | { kind: 'cycle-tab'; dir: 1 | -1 }
  | { kind: 'preview' }
  | { kind: 'rename' }
  | { kind: 'commit-rename' }
  | { kind: 'cancel-rename' }
  | { kind: 'pass' }

const PASS: SessionPickerAction = { kind: 'pass' }

/**
 * Route one key press. While RENAMING, the rename input owns every key except
 * Enter (commit) and Esc/Ctrl+C (cancel rename — NOT close). Otherwise:
 * Esc/Ctrl+C close, Enter resumes, ↑↓ (or Ctrl+P/N) move, Tab/Shift+Tab cycle
 * tabs, ←/→ cycle only on an empty query (with text they stay cursor moves),
 * Space toggles the preview (it never types — fuzzy terms don't need literal
 * spaces), Ctrl+R starts the inline rename. Everything else belongs to the
 * focused search input.
 */
export function routeSessionPickerKey(
  name: string,
  mods: { ctrl?: boolean; shift?: boolean },
  ctx: SessionPickerKeyContext
): SessionPickerAction {
  if (ctx.renaming) {
    if (name === 'return') return { kind: 'commit-rename' }
    if (name === 'escape' || (mods.ctrl && name === 'c')) return { kind: 'cancel-rename' }
    return PASS
  }
  if (name === 'escape' || (mods.ctrl && name === 'c')) return { kind: 'close' }
  if (name === 'return') return { kind: 'resume' }
  if (name === 'up' || (mods.ctrl && name === 'p')) return { kind: 'move', dir: -1 }
  if (name === 'down' || (mods.ctrl && name === 'n')) return { kind: 'move', dir: 1 }
  if (name === 'tab') return { kind: 'cycle-tab', dir: mods.shift ? -1 : 1 }
  if ((name === 'left' || name === 'right') && ctx.queryEmpty) {
    return { kind: 'cycle-tab', dir: name === 'left' ? -1 : 1 }
  }
  if (name === 'space') return { kind: 'preview' }
  if (mods.ctrl && name === 'r') return { kind: 'rename' }
  return PASS
}

// ── relative time + row meta ──────────────────────────────────────────────

/** Epoch seconds OR milliseconds → ms (DB rows are seconds; be lenient). */
function toMs(epoch: number): number {
  return epoch >= 1e12 ? epoch : epoch * 1000
}

const TIME_STEPS: ReadonlyArray<{ ms: number; unit: string }> = [
  { ms: 60_000, unit: 'minute' },
  { ms: 3_600_000, unit: 'hour' },
  { ms: 86_400_000, unit: 'day' },
  { ms: 604_800_000, unit: 'week' },
  { ms: 2_629_800_000, unit: 'month' },
  { ms: 31_557_600_000, unit: 'year' }
]

/** "just now" / "1 minute ago" / "5 hours ago" / "2 weeks ago" … */
export function relativeTime(epoch: number | undefined, nowMs: number): string {
  if (!epoch) return 'unknown'
  const delta = nowMs - toMs(epoch)
  if (delta < 60_000) return 'just now'
  for (let i = TIME_STEPS.length - 1; i >= 0; i--) {
    const step = TIME_STEPS[i]
    if (step && delta >= step.ms) {
      const n = Math.floor(delta / step.ms)
      return `${n} ${step.unit}${n === 1 ? '' : 's'} ago`
    }
  }
  return 'just now'
}

/** Tail-truncate a path-ish string to `max` chars (`…tail/of/path`). */
export function tailTruncate(text: string, max: number): string {
  if (text.length <= max) return text
  return `…${text.slice(text.length - (max - 1))}`
}

/** Max cwd tail shown in a row's meta line. */
const META_CWD_MAX = 40

/** Row meta line: relative time · source · N msgs · cwd (when present). */
export function rowMeta(row: SessionRow, nowMs: number): string {
  const parts = [
    relativeTime(row.lastActive || row.startedAt, nowMs),
    row.source || 'unknown',
    `${row.messageCount} msgs`
  ]
  if (row.cwd) parts.push(tailTruncate(row.cwd, META_CWD_MAX))
  return parts.join(' · ')
}

// ── slash entry points ────────────────────────────────────────────────────

/** Parse a `/sessions <tab>` argument (case-insensitive, singular tolerated;
 *  bare/empty → the default Recent tab). Garbage → undefined (usage notice). */
export function parseSessionTabArg(arg: string): SessionTabId | undefined {
  const a = arg.trim().toLowerCase()
  if (!a || a === 'recent') return 'recent'
  if (a === 'cron') return 'cron'
  if (a === 'gateway' || a === 'gateways') return 'gateways'
  if (a === 'all') return 'all'
  return undefined
}

/**
 * Resolve a `/resume <id|name>` argument against listed rows (the direct
 * path): exact id → unique id prefix → exact title (case-insensitive) →
 * unique case-insensitive title substring. Ambiguous/missing → undefined.
 */
export function resolveSessionArg(rows: readonly SessionRow[], arg: string): SessionRow | undefined {
  const needle = arg.trim()
  if (!needle) return undefined
  const exactId = rows.find(r => r.id === needle)
  if (exactId) return exactId
  const idPrefix = rows.filter(r => r.id.startsWith(needle))
  if (idPrefix.length === 1) return idPrefix[0]
  const lower = needle.toLowerCase()
  const exactTitle = rows.filter(r => r.title.toLowerCase() === lower)
  if (exactTitle.length === 1) return exactTitle[0]
  const sub = rows.filter(r => r.title.toLowerCase().includes(lower))
  if (sub.length === 1) return sub[0]
  return undefined
}
