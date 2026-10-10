import { writeFileSync } from 'node:fs'

import type { ScrollBoxHandle } from '@hermes/ink'
import { evictInkCaches } from '@hermes/ink'
import type { InflightTurn, SessionResumeResult, Usage } from '@hermes/shared/gateway-events'
import { type RefObject, useCallback, useEffect, useMemo, useRef } from 'react'

import { STARTUP_WORKSPACE_CWD } from '../config/env.js'
import { buildSetupRequiredSections, setupRequiredTitle } from '../content/setup.js'
import { introMsg, toTranscriptMessages } from '../domain/messages.js'
import { ZERO } from '../domain/usage.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  SessionActivateResponse,
  SessionCloseResponse,
  SessionCreateResponse,
  SessionTitleResponse,
  SetupStatusResponse
} from '../gatewayTypes.js'
import { t } from '../i18n/runtime.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Msg, PanelSection, SessionInfo } from '../types.js'

import { applyConnectionRequest, clearConnectionOperation } from './connectionOperationStore.js'
import type { ComposerActions, GatewayRpc, StateSetter } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { scheduleResumeScrollToBottom } from './sessionResumeView.js'
import { turnController } from './turnController.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'
import { describeCredentialWarning } from './userMessages.js'

export { refreshSessionView, scheduleResumeScrollToBottom } from './sessionResumeView.js'

const usageFrom = (info: null | SessionInfo): Usage => (info?.usage ? { ...ZERO, ...info.usage } : ZERO)

const statusFromLiveSession = (status?: string, running = false) => {
  if (status === 'waiting') {
    return t('session.status.waitingForInput')
  }

  if (status === 'starting') {
    return t('session.status.startingAgent')
  }

  return running || status === 'working' ? 'running…' : 'ready'
}

export const writeActiveSessionFile = (sessionId: null | string, file = process.env.HERMES_TUI_ACTIVE_SESSION_FILE) => {
  if (!file || !sessionId) {
    return
  }

  try {
    writeFileSync(file, JSON.stringify({ session_id: sessionId }), { mode: 0o600 })
  } catch {
    // Best-effort shell epilogue hint only; never break live session changes.
  }
}

export const liveSessionInflightMessages = (inflight?: null | InflightTurn): Msg[] => {
  const user = String(inflight?.user ?? '').trim()

  return user
    ? toTranscriptMessages([
        {
          role: 'user',
          text: user,
          ...(inflight?.display_kind ? { display_kind: inflight.display_kind } : {}),
          ...(inflight?.display_metadata ? { display_metadata: inflight.display_metadata } : {})
        }
      ])
    : []
}

export const hydrateLiveSessionInflight = (inflight?: null | InflightTurn) => {
  const assistant = String(inflight?.assistant ?? '')

  if (!assistant && !inflight?.streaming) {
    return
  }

  turnController.hydrateStreamingText(assistant)
}

export const signalFreshSessionBoundary = (
  previousSid: null | string,
  nextSid: null | string,
  onFreshSessionStarted?: (sessionId: string) => void
) => {
  if (!previousSid || !nextSid || previousSid === nextSid || !onFreshSessionStarted) {
    return false
  }

  onFreshSessionStarted(nextSid)

  return true
}

const trimTail = (items: Msg[]) => {
  const q = [...items]

  while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') {
    q.pop()
  }

  if (q.at(-1)?.role === 'user') {
    q.pop()
  }

  return q
}

export interface UseSessionLifecycleOptions {
  colsRef: { current: number }
  composerActions: ComposerActions
  gw: GatewayClient
  onFreshSessionStarted?: (sessionId: string) => void
  panel: (title: string, sections: PanelSection[]) => void
  rpc: GatewayRpc
  scrollRef: RefObject<null | ScrollBoxHandle>
  setHistoryItems: StateSetter<Msg[]>
  setLastUserMsg: StateSetter<string>
  setSessionStartedAt: StateSetter<number>
  setStickyPrompt: StateSetter<string>
  setVoiceProcessing: StateSetter<boolean>
  setVoiceRecording: StateSetter<boolean>
  sys: (text: string) => void
}

export function useSessionLifecycle(opts: UseSessionLifecycleOptions) {
  const {
    colsRef,
    composerActions,
    gw,
    onFreshSessionStarted,
    panel,
    rpc,
    scrollRef,
    setHistoryItems,
    setLastUserMsg,
    setSessionStartedAt,
    setStickyPrompt,
    setVoiceProcessing,
    setVoiceRecording,
    sys
  } = opts

  // Plugin on_session_finalize text comes back on the close result and is shown as system lines (never a
  // model turn). `deferMessages`: the caller resets the transcript next and shows them itself afterwards.
  const closeSession = useCallback(
    async (targetSid?: null | string, deferMessages = false) => {
      const closed = targetSid ? await rpc<SessionCloseResponse>('session.close', { session_id: targetSid }) : null

      if (!deferMessages) {
        closed?.messages?.forEach(message => sys(message))
      }

      return closed
    },
    [rpc, sys]
  )

  const cancelResumeScrollRef = useRef<null | (() => void)>(null)

  const resetSession = useCallback(() => {
    cancelResumeScrollRef.current?.()
    cancelResumeScrollRef.current = null
    turnController.fullReset()
    setVoiceRecording(false)
    setVoiceProcessing(false)
    patchUiState({ bgTasks: new Set(), info: null, sid: null, storedSid: null, usage: ZERO })
    setHistoryItems([])
    setLastUserMsg('')
    setStickyPrompt('')
    composerActions.setComposerTokens([])
    // Half-prune: new session has new keys, but keep a warm pool in case
    // the user resumes back to the prior session.
    evictInkCaches('half')
  }, [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt, setVoiceProcessing, setVoiceRecording])

  useEffect(
    () => () => {
      cancelResumeScrollRef.current?.()
      cancelResumeScrollRef.current = null
    },
    []
  )

  const resetVisibleHistory = useCallback(
    (info: null | SessionInfo = null) => {
      turnController.idle()
      turnController.clearReasoning()
      turnController.turnTools = []
      turnController.persistedToolLabels.clear()

      setHistoryItems(info ? [introMsg(info)] : [])
      setStickyPrompt('')
      setLastUserMsg('')
      composerActions.setComposerTokens([])
      patchTurnState({ activity: [] })
      patchUiState({ info, usage: usageFrom(info) })
    },
    [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt]
  )

  const startNewSession = useCallback(
    async (msg?: string, title?: string, keepCurrent = false) => {
      const setup = await rpc<SetupStatusResponse>('setup.status', {})

      if (setup?.provider_configured === false) {
        panel(setupRequiredTitle(), buildSetupRequiredSections())
        patchUiState({ status: t('session.status.setupRequired') })

        return null
      }

      const previousSid = getUiState().sid
      const closed = keepCurrent ? null : await closeSession(previousSid, true)

      const r = await rpc<SessionCreateResponse>('session.create', {
        cols: colsRef.current,
        ...(STARTUP_WORKSPACE_CWD ? { cwd: STARTUP_WORKSPACE_CWD } : {})
      })

      if (!r) {
        patchUiState({ status: 'ready' })

        return null
      }

      // The durable id lives on the create result; the lazy-create `info` does
      // not carry it, and session.resume / the exit epilogue need the stored id.
      const storedSid = r.stored_session_id || r.session_id
      const info = r.info ? { ...r.info, stored_session_id: storedSid } : null
      const requestedTitle = title?.trim() ?? ''

      resetSession()
      setSessionStartedAt(Date.now())

      writeActiveSessionFile(storedSid)
      patchUiState({
        info,
        sid: r.session_id,
        status: info?.version ? 'ready' : t('session.status.startingAgent'),
        storedSid,
        usage: usageFrom(info)
      })

      if (info) {
        setHistoryItems([introMsg(info)])
      }

      if (info?.credential_warning) {
        sys(`warning: ${describeCredentialWarning(info.credential_warning)}`)
      }

      if (info?.config_warning) {
        sys(`warning: ${info.config_warning}`)
      }

      if (msg) {
        sys(msg)
      }

      // After the reset above, so the closed session's plugin messages stay visible.
      closed?.messages?.forEach(message => sys(message))

      if (requestedTitle) {
        rpc<SessionTitleResponse>('session.title', {
          session_id: r.session_id,
          title: requestedTitle
        })
          .then(result => {
            if (!result || getUiState().sid !== r.session_id) {
              return
            }

            const nextTitle = (result.title ?? requestedTitle).trim()
            const suffix = result.pending ? t('session.lifecycle.titleQueuedSuffix') : ''
            patchUiState({ sessionTitle: nextTitle })
            sys(`${t('session.lifecycle.sessionTitleSet', nextTitle)}${suffix}`)
          })
          .catch((err: unknown) => {
            if (getUiState().sid !== r.session_id) {
              return
            }

            const message = err instanceof Error ? err.message : String(err)
            sys(`warning: ${t('session.lifecycle.failedToSetTitle', message)}`)
          })
      }

      signalFreshSessionBoundary(previousSid, r.session_id, onFreshSessionStarted)

      return r.session_id
    },
    [closeSession, colsRef, onFreshSessionStarted, panel, resetSession, rpc, setHistoryItems, setSessionStartedAt, sys]
  )

  const newSession = useCallback(
    (msg?: string, title?: string) => startNewSession(msg, title, false),
    [startNewSession]
  )

  const newLiveSession = useCallback(
    (msg = t('session.lifecycle.newLiveSessionStarted'), title?: string) => {
      patchOverlayState({ sessions: false })

      return startNewSession(msg, title, true)
    },
    [startNewSession]
  )

  const activateLiveSession = useCallback(
    (id: string) => {
      patchOverlayState({ sessions: false })
      patchUiState({ status: t('session.status.switchingSession') })
      // The card belongs to the session being left; the activated one answers with its own.
      clearConnectionOperation()

      gw.request<SessionActivateResponse>('session.activate', { session_id: id })
        .then(raw => {
          const r = asRpcResult<SessionActivateResponse>(raw)

          if (!r) {
            sys(`error: ${t('session.common.invalidResponse', 'session.activate')}`)

            return patchUiState({ status: 'ready' })
          }

          const info = r.info ?? null
          // Agent-less (lazy) activations answer with `_fallback_session_info`, which
          // has no stored_session_id; the durable id is the response's session_key.
          const storedSid = r.session_key || r.session_id
          const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')

          resetSession()
          setSessionStartedAt(r.started_at ? r.started_at * 1000 : Date.now())
          const transcript = [...toTranscriptMessages(r.messages), ...liveSessionInflightMessages(r.inflight)]
          setHistoryItems(info ? [introMsg(info), ...transcript] : transcript)
          writeActiveSessionFile(storedSid)
          patchUiState({
            busy: running,
            info,
            sid: r.session_id,
            status: statusFromLiveSession(r.status, running),
            storedSid,
            usage: usageFrom(info)
          })
          hydrateLiveSessionInflight(r.inflight)

          if (r.pending_connection) {
            applyConnectionRequest(r.pending_connection)
          }

          cancelResumeScrollRef.current?.()
          cancelResumeScrollRef.current = scheduleResumeScrollToBottom(scrollRef)
        })
        .catch((e: Error) => {
          sys(`error: ${e.message}`)
          patchUiState({ status: 'ready' })
        })
    },
    [gw, resetSession, scrollRef, setHistoryItems, setSessionStartedAt, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      patchOverlayState({ sessions: false })
      patchUiState({ status: t('session.status.resuming') })

      return rpc<SetupStatusResponse>('setup.status', {}).then(setup => {
        if (setup?.provider_configured === false) {
          panel(setupRequiredTitle(), buildSetupRequiredSections())
          patchUiState({ status: t('session.status.setupRequired') })

          return
        }

        return gw
          .request<SessionResumeResult>('session.resume', { cols: colsRef.current, session_id: id })
          .then(raw => {
            const r = asRpcResult<SessionResumeResult>(raw)

            if (!r) {
              sys(`error: ${t('session.common.invalidResponse', 'session.resume')}`)

              return patchUiState({ status: 'ready' })
            }

            const storedSid = r.info?.stored_session_id || r.stored_session_id || r.resumed || id
            const info = r.info ? { ...r.info, stored_session_id: storedSid } : null

            const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')

            resetSession()
            setSessionStartedAt(r.started_at ? r.started_at * 1000 : Date.now())

            const resumed = [...toTranscriptMessages(r.messages), ...liveSessionInflightMessages(r.inflight)]

            setHistoryItems(info ? [introMsg(info), ...resumed] : resumed)
            writeActiveSessionFile(storedSid)
            patchUiState({
              busy: running,
              info,
              sid: r.session_id,
              status: statusFromLiveSession(r.status ?? undefined, running),
              storedSid,
              usage: usageFrom(info)
            })
            hydrateLiveSessionInflight(r.inflight)

            if (r.pending_connection) {
              applyConnectionRequest(r.pending_connection)
            } else {
              clearConnectionOperation()
            }

            cancelResumeScrollRef.current?.()
            cancelResumeScrollRef.current = scheduleResumeScrollToBottom(scrollRef)
          })
          .catch((e: Error) => {
            sys(`error: ${e.message}`)
            patchUiState({ status: 'ready' })
          })
      })
    },
    [colsRef, gw, panel, resetSession, rpc, scrollRef, setHistoryItems, setSessionStartedAt, sys]
  )

  const guardBusySessionSwitch = useCallback(
    (what = t('session.lifecycle.switchSessions')) => {
      if (!getUiState().busy) {
        return false
      }

      sys(t('session.lifecycle.interruptBeforeSwitch', what))

      return true
    },
    [sys]
  )

  return useMemo(
    () => ({
      activateLiveSession,
      closeSession,
      guardBusySessionSwitch,
      newLiveSession,
      newSession,
      resetSession,
      resetVisibleHistory,
      resumeById,
      trimLastExchange: trimTail
    }),
    [
      activateLiveSession,
      closeSession,
      guardBusySessionSwitch,
      newLiveSession,
      newSession,
      resetSession,
      resetVisibleHistory,
      resumeById,
      trimTail
    ]
  )
}
