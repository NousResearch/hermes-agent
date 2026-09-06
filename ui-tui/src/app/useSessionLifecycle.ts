import { writeFileSync } from 'node:fs'

import type { ScrollBoxHandle } from '@hermes/ink'
import { evictInkCaches } from '@hermes/ink'
import { type RefObject, useCallback, useEffect, useMemo, useRef } from 'react'

import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js'
import { introMsg, toTranscriptMessages } from '../domain/messages.js'
import { ZERO } from '../domain/usage.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  SessionActivateResponse,
  SessionCloseResponse,
  SessionCreateResponse,
  SessionInflightTurn,
  SessionResumeResponse,
  SessionTitleResponse,
  SetupStatusResponse
} from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Msg, PanelSection, SessionInfo, Usage, UserInputReq } from '../types.js'

import type { ComposerActions, GatewayRpc, StateSetter } from './interfaces.js'
import { getOverlayState, patchOverlayState } from './overlayStore.js'
import { scheduleResumeScrollToBottom } from './sessionResumeView.js'
import { turnController } from './turnController.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

export { refreshSessionView, scheduleResumeScrollToBottom } from './sessionResumeView.js'

const usageFrom = (info: null | SessionInfo): Usage => (info?.usage ? { ...ZERO, ...info.usage } : ZERO)

const statusFromLiveSession = (status?: string, running = false) => {
  if (status === 'waiting') {
    return 'waiting for input…'
  }

  if (status === 'starting') {
    return 'starting agent…'
  }

  return running || status === 'working' ? 'running…' : 'ready'
}

export const pendingUserInputFromResponse = (raw: unknown, sessionId: string): UserInputReq | null => {
  const payload = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  const rows = Array.isArray(payload.requests) ? payload.requests : Array.isArray(payload.data) ? payload.data : []
  const key = String(sessionId || '').trim()

  for (const row of rows) {
    if (!row || typeof row !== 'object') {continue}
    const value = row as Record<string, unknown>

    if (String(value.status ?? 'pending').trim().toLowerCase() !== 'pending') {continue}
    const requestId = String(value.request_id ?? value.requestId ?? '').trim()
    const rowSessionId = String(value.session_id ?? value.sessionId ?? key).trim()

    if (!requestId || !key || rowSessionId !== key || !Array.isArray(value.questions)) {continue}

    const questions = value.questions
      .slice(0, 5)
      .filter(question => question && typeof question === 'object')
      .map(question => {
        const item = question as Record<string, unknown>
        const id = String(item.id ?? '').trim()
        const text = String(item.text ?? item.question ?? '').trim()

        const options = Array.isArray(item.options)
          ? item.options.slice(0, 4).filter(option => typeof option === 'string' && option.trim()).map(option => option.trim())
          : []

        const defaultValue = item.default

        return {
          allowFreeText: item.allow_free_text === true || item.allowFreeText === true,
          ...(typeof defaultValue === 'string' || typeof defaultValue === 'number' || typeof defaultValue === 'boolean'
            ? { defaultValue: String(defaultValue) }
            : {}),
          id,
          options,
          text
        }
      })
      .filter(question => question.id && question.text)

    if (!questions.length) {continue}

    return {
      context: String(value.context ?? '').trim(),
      expiresAt: Number.isFinite(value.expires_at) ? Number(value.expires_at) : 0,
      questions,
      requestId,
      sessionId: key
    }
  }

  return null
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

export const liveSessionInflightMessages = (inflight?: null | SessionInflightTurn): Msg[] => {
  const user = String(inflight?.user ?? '').trim()

  return user ? [{ role: 'user', text: user }] : []
}

export const hydrateLiveSessionInflight = (inflight?: null | SessionInflightTurn) => {
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

  const closeSession = useCallback(
    (targetSid?: null | string) =>
      targetSid ? rpc<SessionCloseResponse>('session.close', { session_id: targetSid }) : Promise.resolve(null),
    [rpc]
  )

  const cancelResumeScrollRef = useRef<null | (() => void)>(null)

  const resetSession = useCallback(() => {
    cancelResumeScrollRef.current?.()
    cancelResumeScrollRef.current = null
    turnController.fullReset()
    setVoiceRecording(false)
    setVoiceProcessing(false)
    patchUiState({ bgTasks: new Set(), info: null, sid: null, usage: ZERO })
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

  const replayPendingUserInput = useCallback(
    async (sessionId: string) => {
      const key = String(sessionId || '').trim()

      if (!key) {return}

      try {
        const raw = await gw.request<Record<string, unknown>>('user_input.pending', { session_id: key })

        if (getUiState().sid !== key) {return}
        const request = pendingUserInputFromResponse(raw, key)

        if (request) {
          patchOverlayState({ userInput: request })
          patchUiState({ status: 'waiting for user input…' })

          return
        }

        if (getOverlayState().userInput?.sessionId === key) {
          patchOverlayState({ userInput: null })
        }
      } catch (error: unknown) {
        if (getUiState().sid === key) {
          sys(`warning: failed to replay user input: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    },
    [gw, sys]
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
        panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
        patchUiState({ status: 'setup required' })

        return null
      }

      const previousSid = getUiState().sid

      if (!keepCurrent) {
        await closeSession(previousSid)
      }

      const r = await rpc<SessionCreateResponse>('session.create', { cols: colsRef.current })

      if (!r) {
        patchUiState({ status: 'ready' })

        return null
      }

      const info = r.info ?? null
      const requestedTitle = title?.trim() ?? ''

      resetSession()
      setSessionStartedAt(Date.now())

      writeActiveSessionFile(r.session_id)
      patchUiState({
        info,
        sid: r.session_id,
        status: info?.version ? 'ready' : 'starting agent…',
        usage: usageFrom(info)
      })

      if (info) {
        setHistoryItems([introMsg(info)])
      }

      if (info?.credential_warning) {
        sys(`warning: ${info.credential_warning}`)
      }

      if (info?.config_warning) {
        sys(`warning: ${info.config_warning}`)
      }

      if (msg) {
        sys(msg)
      }

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
            const suffix = result.pending ? ' (queued while session initializes)' : ''
            patchUiState({ sessionTitle: nextTitle })
            sys(`session title set: ${nextTitle}${suffix}`)
          })
          .catch((err: unknown) => {
            if (getUiState().sid !== r.session_id) {
              return
            }

            const message = err instanceof Error ? err.message : String(err)
            sys(`warning: failed to set session title: ${message}`)
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
    (msg = 'new live session started', title?: string) => {
      patchOverlayState({ sessions: false })

      return startNewSession(msg, title, true)
    },
    [startNewSession]
  )

  const activateLiveSession = useCallback(
    (id: string) => {
      patchOverlayState({ sessions: false })
      patchUiState({ status: 'switching session…' })

      gw.request<SessionActivateResponse>('session.activate', { session_id: id })
        .then(raw => {
          const r = asRpcResult<SessionActivateResponse>(raw)

          if (!r) {
            sys('error: invalid response: session.activate')

            return patchUiState({ status: 'ready' })
          }

          const info = r.info ?? null
          const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')

          resetSession()
          setSessionStartedAt(r.started_at ? r.started_at * 1000 : Date.now())
          const transcript = [...toTranscriptMessages(r.messages), ...liveSessionInflightMessages(r.inflight)]
          setHistoryItems(info ? [introMsg(info), ...transcript] : transcript)
          writeActiveSessionFile(r.session_key ?? r.session_id)
          patchUiState({
            busy: running,
            info,
            sid: r.session_id,
            status: statusFromLiveSession(r.status, running),
            usage: usageFrom(info)
          })
          hydrateLiveSessionInflight(r.inflight)
          void replayPendingUserInput(r.session_id)
          cancelResumeScrollRef.current?.()
          cancelResumeScrollRef.current = scheduleResumeScrollToBottom(scrollRef)
        })
        .catch((e: Error) => {
          sys(`error: ${e.message}`)
          patchUiState({ status: 'ready' })
        })
    },
    [gw, replayPendingUserInput, resetSession, scrollRef, setHistoryItems, setSessionStartedAt, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      patchOverlayState({ sessions: false })
      patchUiState({ status: 'resuming…' })

      rpc<SetupStatusResponse>('setup.status', {}).then(setup => {
        if (setup?.provider_configured === false) {
          panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
          patchUiState({ status: 'setup required' })

          return
        }

        const previousSid = getUiState().sid

        gw.request<SessionResumeResponse>('session.resume', { cols: colsRef.current, session_id: id })
          .then(raw => {
            const r = asRpcResult<SessionResumeResponse>(raw)

            if (!r) {
              sys('error: invalid response: session.resume')

              return patchUiState({ status: 'ready' })
            }

            const info = r.info ?? null
            const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')

            resetSession()
            setSessionStartedAt(r.started_at ? r.started_at * 1000 : Date.now())

            const resumed = [...toTranscriptMessages(r.messages), ...liveSessionInflightMessages(r.inflight)]

            setHistoryItems(info ? [introMsg(info), ...resumed] : resumed)
            writeActiveSessionFile(r.resumed ?? r.session_id)
            patchUiState({
              busy: running,
              info,
              sid: r.session_id,
              status: statusFromLiveSession(r.status, running),
              usage: usageFrom(info)
            })
            hydrateLiveSessionInflight(r.inflight)
            void replayPendingUserInput(r.session_id)
            cancelResumeScrollRef.current?.()
            cancelResumeScrollRef.current = scheduleResumeScrollToBottom(scrollRef)

            if (previousSid && previousSid !== r.session_id) {
              void closeSession(previousSid)
            }
          })
          .catch((e: Error) => {
            sys(`error: ${e.message}`)
            patchUiState({ status: 'ready' })
          })
      })
    },
    [closeSession, colsRef, gw, panel, replayPendingUserInput, resetSession, rpc, scrollRef, setHistoryItems, setSessionStartedAt, sys]
  )

  const guardBusySessionSwitch = useCallback(
    (what = 'switch sessions') => {
      if (!getUiState().busy) {
        return false
      }

      sys(`interrupt the current turn before trying to ${what}`)

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
