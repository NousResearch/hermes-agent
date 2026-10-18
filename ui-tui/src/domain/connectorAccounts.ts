import type { ConnectorAccountRow } from '@hermes/shared/gateway-events'

/** The account-name format the gateway enforces (`connector_alias`). */
export const ACCOUNT_NAME_RE = /^[a-z0-9][a-z0-9-]{0,31}$/

export const isValidAccountName = (name: string): boolean => ACCOUNT_NAME_RE.test(name)

/** An account a reconnect replaced: kept and listed, never selectable. Older backends omit the key. */
export const isRetired = (row: ConnectorAccountRow): boolean => Boolean(row.disabled)

/** A reconnect is a repair: only an account that stopped working offers it. A healthy account
 *  would only be signed in again, so the panel answers that it is connected instead. */
export const needsRepair = (row: ConnectorAccountRow): boolean => !isRetired(row) && row.status !== 'active'

/** Why the panel will not start a reconnect on `row`, or null when it should. */
export function reconnectRefusal(row: ConnectorAccountRow): 'healthy' | 'retired' | null {
  return isRetired(row) ? 'retired' : needsRepair(row) ? null : 'healthy'
}

/** Whether the panel's hint offers `c`: only for a selected row that needs repair (or no selection). */
export const offersReconnect = (row: ConnectorAccountRow | undefined): boolean => !row || needsRepair(row)

/** The name a row is addressed by: its alias, else the app's own account label. */
export const accountName = (row: ConnectorAccountRow): string => row.alias || row.label

export interface AccountView {
  /** Visible rows in display order: by app, then name. */
  rows: ConnectorAccountRow[]
  retired: number
}

const byAppThenName = (a: ConnectorAccountRow, b: ConnectorAccountRow): number =>
  a.connector.localeCompare(b.connector) || accountName(a).localeCompare(accountName(b))

/** Retired accounts are counted and folded unless `showRetired`; they always sort last. */
export function accountView(rows: readonly ConnectorAccountRow[], showRetired: boolean): AccountView {
  const live = rows.filter(row => !isRetired(row)).sort(byAppThenName)
  const retired = rows.filter(isRetired).sort(byAppThenName)

  return { retired: retired.length, rows: showRetired ? [...live, ...retired] : live }
}

/** `app · name · status`, with the vendor label kept for an unnamed account. */
export function accountRowText(row: ConnectorAccountRow, statusLabel: string, unnamedTag: string): string {
  const name = row.alias ? row.alias : `${row.label} ${unnamedTag}`

  return `${row.connector} · ${name} · ${statusLabel}`
}

export type PanelShortcut = 'add' | 'close' | 'reconnect' | 'remove' | 'rename'

const PANEL_SHORTCUTS: Readonly<Record<string, PanelShortcut>> = {
  a: 'add',
  c: 'reconnect',
  q: 'close',
  r: 'rename',
  x: 'remove'
}

/** The panel's letter shortcut for a key press. A Ctrl or Meta chord is never one: Ink reports Ctrl+C as `c`
 *  with `ctrl` set, and that press belongs to the global cancel, not to Reconnect. */
export function panelShortcut(ch: string, key: { ctrl?: boolean; meta?: boolean }): PanelShortcut | null {
  return key.ctrl || key.meta ? null : (PANEL_SHORTCUTS[ch] ?? null)
}

export type NameProblem = 'invalid' | 'taken' | null

/** Client-side check before a rename or an add; the gateway re-checks and answers alias_taken. */
export function nameProblem(
  name: string,
  connector: string,
  rows: readonly ConnectorAccountRow[],
  self?: string
): NameProblem {
  if (!isValidAccountName(name)) {
    return 'invalid'
  }

  const taken = rows.some(
    row => row.connector === connector && row.alias === name && row.connection_id !== self && !isRetired(row)
  )

  return taken ? 'taken' : null
}
