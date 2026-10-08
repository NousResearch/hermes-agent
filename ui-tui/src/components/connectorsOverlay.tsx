import { Box, Text, useInput, useStdout } from '@hermes/ink'
import type {
  ConnectionOperationStatus,
  ConnectionOperationTarget,
  ConnectorAccountRow,
  ConnectorAccountsResult,
  ConnectorsConnectResult,
  ConnectorsListResult
} from '@hermes/shared/gateway-events'
import { JsonRpcGatewayError } from '@hermes/shared/json-rpc-channel'
import { useEffect, useState } from 'react'

import {
  accountName,
  accountRowText,
  accountView,
  isRetired,
  nameProblem,
  type NameProblem
} from '../domain/connectorAccounts.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { Translations } from '../i18n/types.js'
import { useT } from '../i18n/useT.js'
import { openExternalUrl } from '../lib/openExternalUrl.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'
import { TextInput } from './textInput.js'

const VISIBLE = 12
const MIN_WIDTH = 44
const MAX_WIDTH = 96
const POLL_MS = 2_000
const ACCOUNT_OWNER = { type: 'account' } as const
// The gateway's answer for a name another account of the same app already holds.
const ALIAS_TAKEN = 4090
const UNKNOWN_OPERATION = 4004
const SETTLED_STATES = new Set(['connected', 'expired', 'failed', 'not_connected', 'skipped'])

const placeholderTarget: ConnectionOperationTarget = {
  action: 'connect',
  kind: 'connector',
  name: '',
  state: 'not_connected'
}

type Stage =
  | { kind: 'addApp'; apps: string[]; idx: number }
  | { kind: 'addName'; app: string; draft: string }
  | { kind: 'confirm'; row: ConnectorAccountRow }
  | {
      kind: 'link'
      app: string
      name: string
      opId: string
      reconnect: boolean
      target: ConnectionOperationTarget | null
    }
  | { kind: 'list' }
  | { kind: 'rename'; draft: string; row: ConnectorAccountRow }

const problemText = (T: Translations, problem: NameProblem): string =>
  problem === 'taken' ? T.connectors.notice.nameTaken : problem === 'invalid' ? T.connectors.notice.nameInvalid : ''

const statusLabel = (T: Translations, row: ConnectorAccountRow): string =>
  isRetired(row) ? T.connectors.status.retired : T.connectors.status[row.status]

const errorText = (T: Translations, error: unknown): string =>
  error instanceof JsonRpcGatewayError && error.code === ALIAS_TAKEN
    ? T.connectors.notice.nameTaken
    : rpcErrorMessage(error)

export function ConnectorsOverlay({ gw, maxWidth, onClose, t }: ConnectorsOverlayProps) {
  const T = useT()
  const [rows, setRows] = useState<ConnectorAccountRow[]>([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [idx, setIdx] = useState(0)
  const [showRetired, setShowRetired] = useState(false)
  const [stage, setStage] = useState<Stage>({ kind: 'list' })
  const [notice, setNotice] = useState('')
  const [err, setErr] = useState('')

  const { stdout } = useStdout()
  const width = clampOverlayWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6)), maxWidth)

  const load = () =>
    gw
      .request<ConnectorAccountsResult>('connectors.accounts', {})
      .then(r => {
        setRows(r?.accounts ?? [])
        setErr('')
      })
      .catch((e: unknown) => setErr(rpcErrorMessage(e)))
      .finally(() => setLoading(false))

  useEffect(() => {
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gw])

  const view = accountView(rows, showRetired)
  const selectedIdx = Math.min(idx, Math.max(0, view.rows.length - 1))
  const selected = view.rows[selectedIdx]

  const run = <R,>(work: () => Promise<R>, done: (result: R) => void) => {
    setBusy(true)
    setNotice('')
    setErr('')
    work()
      .then(done)
      .catch((e: unknown) => setErr(errorText(T, e)))
      .finally(() => setBusy(false))
  }

  // `name` is what the user sees; only a named account is addressed by alias.
  const startConnect = (app: string, alias: null | string, reconnect: boolean, name = alias ?? app) =>
    run(
      () =>
        gw.request<ConnectorsConnectResult>('connectors.connect', {
          ...(alias ? { alias } : {}),
          connectors: [app],
          owner: ACCOUNT_OWNER,
          reconnect
        }),
      r => {
        const target = r.targets[0] ?? null

        // A healthy account answers `connected` at once and its operation is already closed.
        if (r.settled || (target && SETTLED_STATES.has(target.state))) {
          return target ? finishLink(target, name) : setStage({ kind: 'list' })
        }

        setStage({ app, kind: 'link', name, opId: r.op_id, reconnect, target })
      }
    )

  // The user named this account, so the notice states the outcome, not the agent's choice.
  const finishLink = (target: ConnectionOperationTarget, name: string) => {
    setStage({ kind: 'list' })
    setNotice(
      target.state === 'connected'
        ? T.connectors.notice.connected(target.name, name)
        : T.connectors.notice.notConnected(target.name)
    )
    void load()
  }

  // Poll the account operation while its link is open; Esc leaves the link valid until it expires.
  const linkOp = stage.kind === 'link' ? stage.opId : ''
  const linkApp = stage.kind === 'link' ? stage.app : ''
  const linkName = stage.kind === 'link' ? stage.name : ''

  useEffect(() => {
    if (!linkOp) {
      return
    }

    let stopped = false
    let timer: NodeJS.Timeout | undefined

    const tick = () =>
      gw
        .request<ConnectionOperationStatus>('connectors.operation.status', { op_id: linkOp, owner: ACCOUNT_OWNER })
        .then(snapshot => {
          const target = snapshot?.targets?.[0] ?? null

          if (stopped) {
            return
          }

          if (snapshot?.settled || (target && SETTLED_STATES.has(target.state))) {
            finishLink(target ?? { ...placeholderTarget, name: linkApp }, linkName)

            return
          }

          setStage(current => (current.kind === 'link' && current.opId === linkOp ? { ...current, target } : current))
          timer = setTimeout(tick, POLL_MS)
        })
        .catch((e: unknown) => {
          if (stopped) {
            return
          }

          // The operation closed between polls (settled or expired): read the accounts instead.
          if (e instanceof JsonRpcGatewayError && e.code === UNKNOWN_OPERATION) {
            setStage({ kind: 'list' })
            void load()

            return
          }

          timer = setTimeout(tick, POLL_MS)
        })

    timer = setTimeout(tick, POLL_MS)

    return () => {
      stopped = true
      clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gw, linkOp])

  const openAddApp = () =>
    run(
      () => gw.request<ConnectorsListResult>('connectors.list', { owner: ACCOUNT_OWNER }),
      r => {
        const apps = [...new Set((r?.connectors ?? []).map(row => row.connector))].sort()

        apps.length ? setStage({ apps, idx: 0, kind: 'addApp' }) : setNotice(T.connectors.add.noApps)
      }
    )

  const submitRename = (row: ConnectorAccountRow, draft: string) => {
    const name = draft.trim()
    const problem = nameProblem(name, row.connector, rows, row.connection_id)

    if (problem) {
      return setErr(problemText(T, problem))
    }

    run(
      () =>
        gw.request<ConnectorAccountRow>('connectors.accounts.rename', {
          alias: name,
          connection_id: row.connection_id
        }),
      () => {
        setStage({ kind: 'list' })
        setNotice(T.connectors.notice.renamed(row.connector, name))
        void load()
      }
    )
  }

  const submitAdd = (app: string, draft: string) => {
    const name = draft.trim()
    const problem = nameProblem(name, app, rows)

    if (problem) {
      return setErr(problemText(T, problem))
    }

    startConnect(app, name, false)
  }

  const remove = (row: ConnectorAccountRow) =>
    run(
      () => gw.request('connectors.accounts.remove', { connection_id: row.connection_id }),
      () => {
        setStage({ kind: 'list' })
        setNotice(T.connectors.notice.removed(row.connector, accountName(row)))
        void load()
      }
    )

  const listKey = (ch: string, key: KeyLike) => {
    if (key.escape || ch === 'q') {
      return onClose()
    }

    if (key.upArrow && selectedIdx > 0) {
      return setIdx(selectedIdx - 1)
    }

    if (key.downArrow && selectedIdx < view.rows.length - 1) {
      return setIdx(selectedIdx + 1)
    }

    if (key.tab) {
      return setShowRetired(value => !value)
    }

    if (ch === 'a') {
      return openAddApp()
    }

    if (!selected) {
      return
    }

    if (ch === 'r') {
      setErr('')

      return setStage({ draft: selected.alias ?? '', kind: 'rename', row: selected })
    }

    if (ch === 'x') {
      return setStage({ kind: 'confirm', row: selected })
    }

    if (ch === 'c') {
      return isRetired(selected)
        ? setNotice(T.connectors.notice.retiredNoReconnect)
        : startConnect(selected.connector, selected.alias ?? null, true, accountName(selected))
    }
  }

  useInput((ch, key) => {
    if (busy) {
      return
    }

    switch (stage.kind) {
      case 'list':
        return listKey(ch, key)

      case 'rename':

      case 'addName':
        // The text field owns every other key; Esc cancels exactly once.
        if (key.escape) {
          setErr('')
          setStage({ kind: 'list' })
        }

        return

      case 'addApp':
        if (key.escape) {
          return setStage({ kind: 'list' })
        }

        if (key.upArrow || key.downArrow) {
          const next = (stage.idx + (key.upArrow ? -1 : 1) + stage.apps.length) % stage.apps.length

          return setStage({ ...stage, idx: next })
        }

        if (key.return) {
          setErr('')

          return setStage({ app: stage.apps[stage.idx]!, draft: '', kind: 'addName' })
        }

        return

      case 'confirm':
        if (ch === 'y' || ch === 'Y') {
          return remove(stage.row)
        }

        return setStage({ kind: 'list' })
      case 'link':
        if (key.escape) {
          setStage({ kind: 'list' })

          return setNotice(T.connectors.notice.stopped)
        }

        if (key.return && stage.target?.connect_url) {
          setNotice(openExternalUrl(stage.target.connect_url) ? '' : T.connectors.notice.browserDidNotOpen)
        }
    }
  })

  const footer = (
    <>
      {err ? <Text color={t.color.error}>{T.connectors.errorLine(err)}</Text> : null}
      {notice ? (
        <Text color={t.color.muted} wrap="wrap">
          {notice}
        </Text>
      ) : null}
    </>
  )

  const title = (
    <Text bold color={t.color.accent}>
      {T.connectors.title}
    </Text>
  )

  if (loading) {
    return <Text color={t.color.muted}>{T.connectors.loading}</Text>
  }

  if (stage.kind === 'rename' || stage.kind === 'addName') {
    const heading =
      stage.kind === 'rename'
        ? T.connectors.rename.title(stage.row.connector, accountName(stage.row))
        : T.connectors.add.name(stage.app)

    return (
      <Box flexDirection="column" width={width}>
        {title}
        <Text color={t.color.text}>{heading}</Text>
        <Box paddingLeft={2}>
          <TextInput
            color={t.color.text}
            columns={Math.max(20, width - 4)}
            focus={!busy}
            onChange={draft =>
              setStage(current =>
                current.kind === 'rename' || current.kind === 'addName' ? { ...current, draft } : current
              )
            }
            onSubmit={draft => (stage.kind === 'rename' ? submitRename(stage.row, draft) : submitAdd(stage.app, draft))}
            value={stage.draft}
          />
        </Box>
        {footer}
        <OverlayHint t={t}>
          {stage.kind === 'rename' ? T.connectors.rename.hint : T.connectors.add.nameHint}
        </OverlayHint>
      </Box>
    )
  }

  if (stage.kind === 'addApp') {
    const { items, offset } = windowItems(stage.apps, stage.idx, VISIBLE)

    return (
      <Box flexDirection="column" width={width}>
        {title}
        <Text color={t.color.text}>{T.connectors.add.pickApp}</Text>
        {items.map((app, i) => (
          <Text color={t.color.muted} {...chipRowProps(t, offset + i === stage.idx)} key={app} wrap="truncate-end">
            {offset + i === stage.idx ? '▸ ' : '  '}
            {app}
          </Text>
        ))}
        {footer}
        <OverlayHint t={t}>{T.connectors.add.pickHint}</OverlayHint>
      </Box>
    )
  }

  if (stage.kind === 'link') {
    const heading = stage.reconnect
      ? T.connectors.link.reconnectTitle(stage.app, stage.name)
      : T.connectors.link.title(stage.app, stage.name)

    return (
      <Box flexDirection="column" width={width}>
        {title}
        <Text color={t.color.text}>{heading}</Text>
        {stage.target?.connect_url ? (
          <Text color={t.color.accent}>{stage.target.connect_url}</Text>
        ) : (
          <Text color={t.color.muted}>{T.connectors.link.starting}</Text>
        )}
        {stage.target?.detail ? <Text color={t.color.error}>{stage.target.detail}</Text> : null}
        <Text color={t.color.muted}>{T.connectors.link.waiting}</Text>
        {footer}
        <OverlayHint t={t}>{T.connectors.link.hint}</OverlayHint>
      </Box>
    )
  }

  const labels = view.rows.map(row => accountRowText(row, statusLabel(T, row), T.connectors.unnamedTag))
  const { items, offset } = windowItems(labels, selectedIdx, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      {title}
      {!view.rows.length ? <Text color={t.color.muted}>{T.connectors.empty}</Text> : null}
      {items.map((label, i) => {
        const active = offset + i === selectedIdx
        const row = view.rows[offset + i]!

        return (
          <Text
            color={isRetired(row) ? t.color.muted : t.color.text}
            {...chipRowProps(t, active)}
            key={row.connection_id}
            wrap="truncate-end"
          >
            {active ? '▸ ' : '  '}
            {label}
          </Text>
        )
      })}
      {view.retired ? (
        <Text color={t.color.muted}>
          {showRetired ? T.connectors.retiredShown(view.retired) : T.connectors.retiredFolded(view.retired)}
        </Text>
      ) : null}
      {stage.kind === 'confirm' ? (
        <Text color={t.color.warn}>{T.connectors.remove.confirm(stage.row.connector, accountName(stage.row))}</Text>
      ) : null}
      {footer}
      <OverlayHint t={t}>{T.connectors.hint}</OverlayHint>
    </Box>
  )
}

interface KeyLike {
  downArrow: boolean
  escape: boolean
  tab: boolean
  upArrow: boolean
}

interface ConnectorsOverlayProps {
  gw: GatewayClient
  maxWidth?: number
  onClose: () => void
  t: Theme
}
