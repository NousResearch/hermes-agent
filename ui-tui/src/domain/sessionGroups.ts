import type { ProjectTreeNode } from '@hermes/shared/gateway-events'

export interface SessionGroup<T extends { id: string }> {
  /** Stable project id; `null` for sessions no project claimed. */
  id: null | string
  label: string
  rows: T[]
}

/**
 * Bucket rows by the project that owns them, mirroring the desktop sidebar's
 * grouping (`projects.tree` is the one authoritative builder). Groups follow
 * the tree's order (most recently active first); row order inside a group is
 * preserved. Rows the tree doesn't claim land in a trailing `unclaimedLabel`
 * group so nothing disappears when the tree is stale or capped.
 */
export function groupSessionsByProject<T extends { id: string }>(
  rows: readonly T[],
  projects: readonly ProjectTreeNode[],
  unclaimedLabel: string
): SessionGroup<T>[] {
  const owner = new Map<string, ProjectTreeNode>()

  for (const project of projects) {
    for (const id of project.sessionIds ?? []) {
      if (!owner.has(id)) {
        owner.set(id, project)
      }
    }
  }

  const byProject = new Map<string, SessionGroup<T>>()
  const unclaimed: T[] = []

  for (const row of rows) {
    const project = owner.get(row.id)

    if (!project) {
      unclaimed.push(row)

      continue
    }

    const group: SessionGroup<T> = byProject.get(project.id) ?? { id: project.id, label: project.label, rows: [] }
    group.rows.push(row)
    byProject.set(project.id, group)
  }

  const ordered = projects.flatMap(p => byProject.get(p.id) ?? [])

  return unclaimed.length ? [...ordered, { id: null, label: unclaimedLabel, rows: unclaimed }] : ordered
}

/** How much of a project group the picker shows: just its name, the first rows, or all. */
export type GroupView = 'all' | 'collapsed' | 'preview'

export const GROUP_PREVIEW_ROWS = 3
export const DEFAULT_GROUP_VIEW: GroupView = 'preview'

export const groupKey = (group: { id: null | string }): string => group.id ?? '__other__'

/** collapsed → preview → all → collapsed; `all` is skipped when it would equal the preview. */
export const nextGroupView = (view: GroupView, rowCount: number): GroupView => {
  if (view === 'collapsed') {
    return 'preview'
  }

  if (view === 'preview') {
    return rowCount > GROUP_PREVIEW_ROWS ? 'all' : 'collapsed'
  }

  return 'collapsed'
}

export const visibleGroupRows = <T>(rows: readonly T[], view: GroupView): readonly T[] =>
  view === 'collapsed' ? [] : view === 'preview' ? rows.slice(0, GROUP_PREVIEW_ROWS) : rows

/** One selectable line of the switcher list (the pinned "+ new" row sits above these). */
export type SwitcherRow<L extends { id: string }, H extends { id: string }> =
  | { group: SessionGroup<H>; kind: 'header'; shown: number; view: GroupView }
  | { item: L; kind: 'live' }
  | { kind: 'history'; row: H }

export const switcherRowKey = (row: SwitcherRow<{ id: string }, { id: string }>): string =>
  row.kind === 'header'
    ? `group:${groupKey(row.group)}`
    : row.kind === 'live'
      ? `live:${row.item.id}`
      : `hist:${row.row.id}`

/**
 * Selectable rows in display order: live sessions, then resumable history. With
 * `groups` the history becomes header rows (selectable, cycle their view) each
 * followed by as many of its sessions as the group's view allows; without, flat.
 */
export function buildSwitcherRows<L extends { id: string }, H extends { id: string }>(
  live: readonly L[],
  history: readonly H[],
  groups: readonly SessionGroup<H>[],
  views: Readonly<Record<string, GroupView>>
): SwitcherRow<L, H>[] {
  const rows: SwitcherRow<L, H>[] = live.map(item => ({ item, kind: 'live' }))

  if (!groups.length) {
    return [...rows, ...history.map((row): SwitcherRow<L, H> => ({ kind: 'history', row }))]
  }

  for (const group of groups) {
    const view = views[groupKey(group)] ?? DEFAULT_GROUP_VIEW
    const shown = visibleGroupRows(group.rows, view)

    rows.push({ group, kind: 'header', shown: shown.length, view })

    for (const row of shown) {
      rows.push({ kind: 'history', row })
    }
  }

  return rows
}

interface Searchable {
  id: string
  preview?: null | string
  title?: null | string
}

export interface SwitcherViewState {
  grouped: boolean
  query: string
  views: Readonly<Record<string, GroupView>>
}

const searchText = (row: Searchable, extra = '') =>
  `${row.title ?? ''} ${row.preview ?? ''} ${row.id} ${extra}`.toLowerCase()

/**
 * `buildSwitcherRows` behind the picker's search box. Every whitespace-separated
 * token must match a session's title / preview / id or its project's label, so
 * "alpha fix" finds the "fix" sessions under project "alpha" and a bare project
 * name keeps the whole project. While searching, matching groups show every
 * match (their collapsed / first-3 view is a browsing aid, not a filter).
 */
export function deriveSwitcherRows<L extends Searchable, H extends Searchable>(
  live: readonly L[],
  history: readonly H[],
  groups: readonly SessionGroup<H>[],
  { grouped, query, views }: SwitcherViewState
): SwitcherRow<L, H>[] {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean)

  if (!tokens.length) {
    return buildSwitcherRows(live, history, grouped ? groups : [], views)
  }

  const hit = (row: Searchable, label = '') => {
    const text = searchText(row, label)

    return tokens.every(token => text.includes(token))
  }

  const labelById = new Map(groups.flatMap(g => g.rows.map((r): [string, string] => [r.id, g.label])))

  const matched = groups.map(g => ({ ...g, rows: g.rows.filter(r => hit(r, g.label)) })).filter(g => g.rows.length)

  return buildSwitcherRows(
    live.filter(r => hit(r)),
    history.filter(r => hit(r, labelById.get(r.id))),
    grouped ? matched : [],
    Object.fromEntries(matched.map((g): [string, GroupView] => [groupKey(g), 'all']))
  )
}
