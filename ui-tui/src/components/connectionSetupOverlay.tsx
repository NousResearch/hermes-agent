import { Box, Text, useInput } from '@hermes/ink'
import type {
  ConnectionOperationTarget,
  ConnectionRespondParams,
  ConnectionRespondResult,
  ConnectionTargetAction,
  ConnectionTargetEnvField,
  ConnectionTargetState,
  ConnectorsConnectResult
} from '@hermes/shared/gateway-events'
import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import type { ConnectionOperationSnapshot } from '../app/connectionOperationStore.js'
import {
  $connectionOperation,
  dismissConnectionOperation,
  isSettledOperation
} from '../app/connectionOperationStore.js'
import { useGateway } from '../app/gatewayContext.js'
import { $uiSessionId } from '../app/uiStore.js'
import type { Translations } from '../i18n/types.js'
import { useT } from '../i18n/useT.js'
import { openExternalUrl } from '../lib/openExternalUrl.js'
import type { Theme } from '../theme.js'

import { TextInput } from './textInput.js'

interface ConnectionSetupOverlayProps {
  cols: number
  t: Theme
}

type Phase = 'authorized' | 'browser' | 'form' | 'retry' | 'working'

interface AnsweredRow {
  name: string
  seq: number
  state: ConnectionTargetState
}

const SENDING_TIMEOUT_MS = 5_000

type SendResult = ConnectionRespondResult | ConnectorsConnectResult

const mayAnswer = (operation: ConnectionOperationSnapshot | null, sid: null | string, busy: boolean): boolean =>
  Boolean(operation) && Boolean(sid) && !busy && !isSettledOperation(operation?.opId ?? '')

const isSending = (operation: ConnectionOperationSnapshot | null, answered: AnsweredRow | null): boolean => {
  if (!answered || !operation || operation.seq > answered.seq) {
    return false
  }

  return operation.targets.find(item => item.name === answered.name)?.state === answered.state
}

interface InputKey {
  downArrow: boolean
  escape: boolean
  leftArrow: boolean
  return: boolean
  rightArrow: boolean
  shift: boolean
  tab: boolean
  upArrow: boolean
}

const RESOLVED_STATES = ['connected', 'skipped', 'not_connected']

const isUnresolved = (target: ConnectionOperationTarget): boolean =>
  !RESOLVED_STATES.includes(target.state) || (target.state === 'connected' && Boolean(target.discovery_error))

// The card's title verb, resolved from the active catalog at render (never at import).
const verbOf = (T: Translations, action: ConnectionTargetAction): string => T.connection.verb[action]

const phaseOf = (target: ConnectionOperationTarget): Phase => {
  if (target.state === 'connected') {
    return 'authorized'
  }

  if (target.state === 'failed' || target.state === 'expired') {
    return target.required_env?.length ? 'form' : 'retry'
  }

  if (target.state === 'initiated') {
    return target.connect_url ? 'browser' : 'working'
  }

  return 'form'
}

const failureLine = (T: Translations, target: ConnectionOperationTarget): string =>
  target.state === 'expired' ? T.connection.failure.expired : T.connection.failure.failed

const hasFailed = (target: ConnectionOperationTarget): boolean =>
  target.state === 'failed' || target.state === 'expired'

const initialDraft = (fields: ConnectionTargetEnvField[]): Record<string, string> =>
  Object.fromEntries(fields.map(field => [field.name, field.secret ? '' : field.default]))

const fieldLabel = (field: ConnectionTargetEnvField): string => field.prompt || field.name

interface HeaderProps {
  more: number
  t: Theme
  target: ConnectionOperationTarget
}

function Header({ more, t, target }: HeaderProps) {
  const T = useT()

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.text}>
        {verbOf(T, target.action)} {target.name}
      </Text>
      {more > 0 ? <Text color={t.color.muted}>{T.connection.header.moreToAnswer(more)}</Text> : null}
      {target.instructions ? (
        <Text color={t.color.muted} wrap="wrap">
          {target.instructions}
        </Text>
      ) : null}
    </Box>
  )
}

interface FieldRowProps {
  cols: number
  draftValue: string
  field: ConnectionTargetEnvField
  focused: boolean
  onChange: (value: string) => void
  onSubmit: () => void
  sending: boolean
  showSet: boolean
  t: Theme
}

function FieldRow({ cols, draftValue, field, focused, onChange, onSubmit, sending, showSet, t }: FieldRowProps) {
  const T = useT()

  return (
    <Box flexDirection="column">
      <Text color={focused ? t.color.accent : t.color.label}>
        {focused ? '▸ ' : '  '}
        {fieldLabel(field)}
        {field.required ? ' *' : ''}
      </Text>
      <Box paddingLeft={2}>
        {showSet ? (
          <Text color={t.color.ok}>{T.connection.field.set}</Text>
        ) : (
          <TextInput
            color={t.color.text}
            columns={Math.max(20, cols - 8)}
            focus={!sending && focused}
            ignoreVerticalArrows
            mask={field.secret ? '*' : undefined}
            onChange={onChange}
            onSubmit={onSubmit}
            value={draftValue}
          />
        )}
      </Box>
    </Box>
  )
}

interface NavHandlers {
  confirm: () => void
  row: number
  rows: number
  selectorFocused: boolean
  setAction: (update: (value: 0 | 1) => 0 | 1) => void
  setFocus: (update: (value: number) => number) => void
}

function handleNavKey(key: InputKey, h: NavHandlers): void {
  const up = key.upArrow && !key.shift
  const down = key.downArrow && !key.shift
  const back = () => (h.row - 1 + h.rows) % h.rows
  const forward = () => (h.row + 1) % h.rows

  if (key.shift && key.tab) {
    h.setFocus(back)
  } else if (key.tab || (down && !h.selectorFocused)) {
    h.setFocus(forward)
  } else if (up && !h.selectorFocused) {
    h.setFocus(back)
  } else if (h.selectorFocused && (key.leftArrow || key.rightArrow || up || down)) {
    h.setAction(value => (value === 0 ? 1 : 0))
  } else if (h.selectorFocused && key.return) {
    h.confirm()
  }
}

interface SelectorProps {
  action: 0 | 1
  focused: boolean
  primary: string
  t: Theme
}

function Selector({ action, focused, primary, t }: SelectorProps) {
  const T = useT()

  return (
    <Text color={focused ? t.color.accent : t.color.muted}>
      {action === 0 ? '▸ ' : '  '}
      {primary}
      {'   '}
      {action === 1 ? '▸ ' : '  '}
      {T.connection.selector.skip}
    </Text>
  )
}

interface PhaseProps {
  more: number
  notice: string
  t: Theme
  target: ConnectionOperationTarget
}

function DetailLine({ t, text }: { t: Theme; text: null | string | undefined }) {
  return text ? (
    <Text color={t.color.error} wrap="wrap">
      {text}
    </Text>
  ) : null
}

function FinishingPhase({ t }: { t: Theme }) {
  const T = useT()

  return (
    <Box flexDirection="column">
      <Text color={t.color.muted}>{T.connection.status.finishing}</Text>
      <Text color={t.color.muted}>{T.connection.hint.finishing}</Text>
    </Box>
  )
}

function AuthorizedPhase({ t, target }: { t: Theme; target: ConnectionOperationTarget }) {
  const T = useT()

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.ok}>
        {T.connection.authorized.title}
      </Text>
      <Text color={t.color.muted}>{target.discovery_error ?? ''}</Text>
      <Text color={t.color.accent}>▸ {T.connection.authorized.continue}</Text>
      <Text color={t.color.muted}>{T.connection.authorized.hint}</Text>
    </Box>
  )
}

function BrowserPhase({ more, notice, t, target }: PhaseProps) {
  const T = useT()

  return (
    <Box flexDirection="column">
      <Header more={more} t={t} target={target} />
      <Text color={t.color.accent}>{target.connect_url}</Text>
      <DetailLine t={t} text={target.detail} />
      <DetailLine t={t} text={notice} />
      <Text color={t.color.muted}>{T.connection.hint.browser}</Text>
    </Box>
  )
}

function WorkingPhase({ more, notice, t, target }: PhaseProps) {
  const T = useT()

  return (
    <Box flexDirection="column">
      <Header more={more} t={t} target={target} />
      <Text color={t.color.muted}>{T.connection.status.working}</Text>
      <DetailLine t={t} text={target.detail} />
      <DetailLine t={t} text={notice} />
      <Text color={t.color.muted}>{T.connection.hint.working}</Text>
    </Box>
  )
}

interface RetryPhaseProps extends PhaseProps {
  action: 0 | 1
  sending: boolean
}

function RetryPhase({ action, more, notice, sending, t, target }: RetryPhaseProps) {
  const T = useT()

  return (
    <Box flexDirection="column">
      <Header more={more} t={t} target={target} />
      <Text color={t.color.muted}>{failureLine(T, target)}</Text>
      <DetailLine t={t} text={target.detail} />
      <Selector action={action} focused primary={T.connection.selector.tryAgain} t={t} />
      <DetailLine t={t} text={notice} />
      {sending ? <Text color={t.color.muted}>{T.connection.status.pending}</Text> : null}
      <Text color={t.color.muted}>{T.connection.hint.retry}</Text>
    </Box>
  )
}

interface FormPhaseProps extends PhaseProps {
  action: 0 | 1
  cols: number
  draft: Record<string, string>
  fields: ConnectionTargetEnvField[]
  missingRequired: ConnectionTargetEnvField | undefined
  onChange: (name: string, value: string) => void
  onFieldSubmit: (index: number) => void
  row: number
  selectorFocused: boolean
  sending: boolean
  submittedSecrets: Set<string>
}

function FormPhase(p: FormPhaseProps) {
  const T = useT()
  const { t, target } = p
  const reopened = hasFailed(target)

  return (
    <Box flexDirection="column">
      <Header more={p.more} t={t} target={target} />
      {reopened ? <Text color={t.color.muted}>{failureLine(T, target)}</Text> : null}
      {reopened ? <DetailLine t={t} text={target.detail} /> : null}
      {p.fields.map((field, index) => (
        <FieldRow
          cols={p.cols}
          draftValue={p.draft[field.name] ?? ''}
          field={field}
          focused={p.row === index}
          key={field.name}
          onChange={value => p.onChange(field.name, value)}
          onSubmit={() => p.onFieldSubmit(index)}
          sending={p.sending}
          showSet={p.submittedSecrets.has(field.name) && p.sending}
          t={t}
        />
      ))}
      {reopened ? null : <DetailLine t={t} text={target.detail} />}
      <Selector action={p.action} focused={p.selectorFocused} primary={verbOf(T, target.action)} t={t} />
      {p.missingRequired ? (
        <Text color={t.color.muted}>{T.connection.field.required(fieldLabel(p.missingRequired))}</Text>
      ) : null}
      <DetailLine t={t} text={p.notice} />
      {p.sending ? <Text color={t.color.muted}>{T.connection.status.pending}</Text> : null}
      <Text color={t.color.muted}>{T.connection.hint.form}</Text>
    </Box>
  )
}

export function ConnectionSetupOverlay({ cols, t }: ConnectionSetupOverlayProps) {
  const T = useT()
  const operation = useStore($connectionOperation)
  const sid = useStore($uiSessionId)
  const { gw } = useGateway()
  const unresolved = operation?.targets.filter(isUnresolved) ?? []
  const target = unresolved[0] ?? null
  const fields = useMemo<ConnectionTargetEnvField[]>(() => target?.required_env ?? [], [target?.required_env])
  const targetKey = `${operation?.opId ?? ''}:${target?.name ?? ''}`
  const [draft, setDraft] = useState<Record<string, string>>(() => initialDraft(fields))
  const [focus, setFocus] = useState(0)
  const [action, setAction] = useState<0 | 1>(0)
  // The row that was answered and the state it was in. A frame for another row of the same
  // operation must not clear it; that row's own next state does.
  const [answered, setAnswered] = useState<AnsweredRow | null>(null)
  const [submittedSecrets, setSubmittedSecrets] = useState<Set<string>>(() => new Set())
  const [notice, setNotice] = useState('')
  // One request at a time, so a held key cannot post the same answer again. It is cleared when the
  // request settles either way, so no card state can end up unable to answer.
  const inFlight = useRef(false)

  // A new target starts clean. Every backend snapshot carries a freshly parsed required_env, so the
  // same target's fields only fill in what the draft lacks: a failed Connect keeps what was typed.
  useEffect(() => {
    setDraft({})
    setFocus(0)
    setAction(0)
    setAnswered(null)
    setSubmittedSecrets(new Set())
    setNotice('')
  }, [targetKey])

  useEffect(() => {
    setDraft(current => ({ ...initialDraft(fields), ...current }))
  }, [fields])

  // The third way out of `sending`: the row gives its control back after the timeout even if no
  // frame ever arrives, so Esc and Skip can never be disabled for good.
  useEffect(() => {
    if (!answered) {
      return
    }

    const timer = setTimeout(() => setAnswered(null), SENDING_TIMEOUT_MS)

    return () => clearTimeout(timer)
  }, [answered])

  useEffect(() => {
    if (target?.state === 'connected') {
      setDraft(current =>
        Object.fromEntries(fields.map(field => [field.name, field.secret ? '' : (current[field.name] ?? '')]))
      )
    }
  }, [fields, target?.state])

  const phase = target ? phaseOf(target) : 'form'
  const sending = isSending(operation, answered)
  const missingRequired = fields.find(field => field.required && !draft[field.name]?.trim())
  const rows = phase === 'form' ? fields.length + 1 : 1
  // A snapshot that drops a field leaves the old focus past the last row; the selector owns it.
  const row = Math.min(focus, rows - 1)
  const selectorFocused = phase === 'retry' || row === fields.length

  const send = (start: () => Promise<SendResult>, failure: string) => {
    inFlight.current = true
    setNotice('')
    void start()
      .catch(() => {
        setAnswered(null)
        setNotice(failure)
      })
      .finally(() => {
        inFlight.current = false
      })
  }

  const respond = (result: ConnectionRespondParams['result'], answeredAt: AnsweredRow | null) => {
    if (!operation || !sid || !mayAnswer(operation, sid, inFlight.current)) {
      return
    }

    setAnswered(answeredAt)
    send(
      () =>
        gw.request<ConnectionRespondResult>('connection.respond', {
          op_id: operation.opId,
          owner: { session_id: sid, type: 'session' },
          result
        }),
      T.connection.notice.answerNotDelivered
    )
  }

  const answeredNow = (): AnsweredRow | null =>
    target && operation ? { name: target.name, seq: operation.seq, state: target.state } : null

  const skip = () => {
    if (!target || sending) {
      return
    }

    respond({ targets: [{ name: target.name, status: 'skipped' }] }, answeredNow())
  }

  const connect = () => {
    if (!target || sending) {
      return
    }

    if (missingRequired) {
      const index = fields.indexOf(missingRequired)

      setFocus(index < 0 ? 0 : index)

      return
    }

    setSubmittedSecrets(new Set(fields.filter(field => field.secret).map(field => field.name)))
    respond({ targets: [{ env: draft, name: target.name, status: 'approved' }] }, answeredNow())
  }

  // Try again on a failed row with no credentials to correct: the desktop's re-mint, on the open
  // operation, with this session as the owner.
  const tryAgain = () => {
    if (!target || !operation || !sid || sending || !mayAnswer(operation, sid, inFlight.current)) {
      return
    }

    setAnswered(answeredNow())
    send(
      () =>
        gw.request<ConnectorsConnectResult>('connectors.connect', {
          connectors: [target.name],
          owner: { session_id: sid, type: 'session' },
          reconnect: true
        }),
      T.connection.notice.restartFailed
    )
  }

  const openLink = () => {
    if (!target?.connect_url) {
      return
    }

    setNotice(openExternalUrl(target.connect_url) ? '' : T.connection.notice.browserDidNotOpen)
  }

  // A single input owner guarantees every key causes exactly one action. Esc skips the row in every
  // phase; Ctrl+C reaches the global handler, which interrupts the turn.
  useInput((_ch, key) => {
    // Every row is answered and the settling frame has not landed. Esc closes the card here, so no
    // silence from the gateway can leave it with no way out. The id is remembered, so the settle
    // that follows still writes its transcript lines and this card cannot return.
    if (!target) {
      if (key.escape && operation) {
        dismissConnectionOperation(operation.opId)
      }

      return
    }

    if (phase === 'authorized') {
      if ((key.escape || key.return) && !sending) {
        respond({ settled_by: 'continue' }, answeredNow())
      }

      return
    }

    if (key.escape) {
      return skip()
    }

    if (phase === 'browser') {
      if (key.return) {
        openLink()
      }

      return
    }

    if (phase === 'working') {
      return
    }

    handleNavKey(key, {
      confirm: () => (action === 1 ? skip() : phase === 'retry' ? tryAgain() : connect()),
      rows,
      selectorFocused,
      setAction,
      setFocus,
      row
    })
  })

  if (!operation) {
    return null
  }

  // Every row is answered; the backend settles the operation and its frame closes the card.
  if (!target) {
    return <FinishingPhase t={t} />
  }

  if (phase === 'authorized') {
    return <AuthorizedPhase t={t} target={target} />
  }

  const shared = { more: unresolved.length - 1, notice, t, target }

  if (phase === 'browser') {
    return <BrowserPhase {...shared} />
  }

  if (phase === 'working') {
    return <WorkingPhase {...shared} />
  }

  if (phase === 'retry') {
    return <RetryPhase {...shared} action={action} sending={sending} />
  }

  return (
    <FormPhase
      {...shared}
      action={action}
      cols={cols}
      draft={draft}
      fields={fields}
      missingRequired={missingRequired}
      onChange={(name, value) => setDraft(current => ({ ...current, [name]: value }))}
      onFieldSubmit={index => setFocus(index === fields.length - 1 ? fields.length : index + 1)}
      row={row}
      selectorFocused={selectorFocused}
      sending={sending}
      submittedSecrets={submittedSecrets}
    />
  )
}
