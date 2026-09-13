import { randomUUID } from 'node:crypto'
import { writeFileSync } from 'node:fs'

import type { ScrollBoxHandle } from '@hermes/ink'
import { evictInkCaches } from '@hermes/ink'
import { type RefObject, useCallback, useEffect, useMemo, useRef } from 'react'

import { localCreationOptions } from '../canonicalGateway.js'
import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js'
import { introMsg, toTranscriptMessages } from '../domain/messages.js'
import { ZERO } from '../domain/usage.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  SessionActivateResponse,
  SessionCloseResponse,
  SessionCreateResponse,
  SessionDetachResponse,
  SessionInflightTurn,
  SessionResumeResponse,
  SessionTitleResponse,
  SetupStatusResponse
} from '../gatewayTypes.js'
import { migratePendingInputs } from '../lib/pendingInputs.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Msg, PanelSection, SessionInfo, Usage } from '../types.js'

import type { ComposerActions, GatewayRpc, StateSetter } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { scheduleResumeScrollToBottom } from './sessionResumeView.js'
import { captureDestination } from './submissionDestination.js'
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
      targetSid && !gw.isCanonical ? rpc<SessionCloseResponse>('session.close', { session_id: targetSid }) : Promise.resolve(null),
    [gw.isCanonical, rpc]
  )

  const cancelResumeScrollRef = useRef<null | (() => void)>(null)
  const attachmentFlight = useRef(0)
  const pendingAttachments = useRef(new Set<number>())
  const canonicalSubscriptions = useRef(new Map<string, string>())
  const canonicalDetachFlights = useRef(new Map<string, Promise<void>>())
  const staleAttachments = useRef(new Map<string, { session_id: string; subscription_id: string }>())

  const detachCanonical = useCallback((sessionId?: null | string, subscriptionId?: null | string) => {
    if (!gw.isCanonical || !sessionId || !subscriptionId) {
      return
    }

    const detach = () => rpc<SessionDetachResponse>('session.detach', {
      session_id: sessionId,
      subscription_id: subscriptionId
    }).then(result => {
      if (result?.detached && canonicalSubscriptions.current.get(sessionId) === subscriptionId) {
        canonicalSubscriptions.current.delete(sessionId)
      }
    }).catch(() => undefined)
    const previous = canonicalDetachFlights.current.get(sessionId)
    const pending = previous ? previous.then(detach) : detach()

    canonicalDetachFlights.current.set(sessionId, pending)
    void pending.then(() => {
      if (canonicalDetachFlights.current.get(sessionId) === pending) {
        canonicalDetachFlights.current.delete(sessionId)
      }
    })
  }, [gw.isCanonical, rpc])

  const disposeStaleAttachments = useCallback(() => {
    // In-flight attaches can share a token before either result is adopted.
    // Decide disposal only after their handlers have adopted (or refused) it.
    if (pendingAttachments.current.size) {return}

    const stale = [...staleAttachments.current.values()]
    staleAttachments.current.clear()

    for (const result of stale) {
      if (canonicalSubscriptions.current.get(result.session_id) !== result.subscription_id) {
        detachCanonical(result.session_id, result.subscription_id)
      }
    }
  }, [detachCanonical])

  const finishAttachment = useCallback((flight: number) => {
    pendingAttachments.current.delete(flight)
    disposeStaleAttachments()
  }, [disposeStaleAttachments])

  const discardStaleAttachment = useCallback((result?: null | { session_id: string; subscription_id?: string }) => {
    if (result?.subscription_id) {
      staleAttachments.current.set(JSON.stringify([result.session_id, result.subscription_id]), {
        session_id: result.session_id, subscription_id: result.subscription_id
      })
    }

    disposeStaleAttachments()
  }, [disposeStaleAttachments])

  const adoptAttachment = useCallback((
    result: { session_id: string; subscription_id?: string },
    previousSid?: null | string,
    previousSubscription?: null | string
  ) => {
    if (!gw.isCanonical || !result.subscription_id) {
      return
    }

    canonicalSubscriptions.current.set(result.session_id, result.subscription_id)

    if (previousSid && previousSubscription
        && (previousSid !== result.session_id || previousSubscription !== result.subscription_id)) {
      detachCanonical(previousSid, previousSubscription)
    }
  }, [detachCanonical, gw.isCanonical])

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
      attachmentFlight.current++
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
      const flight = ++attachmentFlight.current
      const previousSid = getUiState().sid
      const previousSubscription = previousSid ? canonicalSubscriptions.current.get(previousSid) : undefined
      pendingAttachments.current.add(flight)

      try {
        const setup = gw.isCanonical ? null : await rpc<SetupStatusResponse>('setup.status', {})

        if (flight !== attachmentFlight.current) {return null}

        if (setup?.provider_configured === false) {
          panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
          patchUiState({ status: 'setup required' })

          return null
        }

        if (!keepCurrent) {
          await closeSession(previousSid)

          if (flight !== attachmentFlight.current) {return null}
        }

        const r = await rpc<SessionCreateResponse>('session.create', gw.isCanonical
          ? { request_id: randomUUID(), ...localCreationOptions() } : { cols: colsRef.current })

        if (flight !== attachmentFlight.current) {
          discardStaleAttachment(r)

          return null
        }

        if (!r) {
          patchUiState({ status: 'ready' })

          return null
        }

        const info = r.info ? { ...r.info, stored_session_id: r.stored_session_id || r.info.stored_session_id } : null
        const requestedTitle = title?.trim() ?? ''

        resetSession()
        setSessionStartedAt(Date.now())

        writeActiveSessionFile(r.session_id)
        patchUiState({
          info,
          sid: r.session_id,
          status: gw.isCanonical || info?.version ? 'ready' : 'starting agent…',
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
        adoptAttachment(r, previousSid, previousSubscription)

        return r.session_id
      } finally {
        finishAttachment(flight)
      }
    },
    [adoptAttachment, closeSession, colsRef, discardStaleAttachment, finishAttachment, gw.isCanonical, onFreshSessionStarted, panel,
      resetSession, rpc, setHistoryItems, setSessionStartedAt, sys]
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
      const flight = ++attachmentFlight.current
      pendingAttachments.current.add(flight)
      const previousSid = getUiState().sid
      const previousSubscription = previousSid ? canonicalSubscriptions.current.get(previousSid) : undefined
      patchOverlayState({ sessions: false })
      patchUiState({ status: 'switching session…' })

      const pendingDetach = canonicalDetachFlights.current.get(id)
      const request = pendingDetach
        ? pendingDetach.then(() => flight === attachmentFlight.current
          ? gw.request<SessionActivateResponse>('session.activate', { session_id: id }) : null)
        : gw.request<SessionActivateResponse>('session.activate', { session_id: id })

      request
        .then(raw => {
          const r = asRpcResult<SessionActivateResponse>(raw)

          if (flight !== attachmentFlight.current) {
            discardStaleAttachment(r)

            return
          }

          if (!r) {
            sys('error: invalid response: session.activate')

            return patchUiState({ status: 'ready' })
          }

          const info = r.info ? { ...r.info, stored_session_id: r.stored_session_id || r.info.stored_session_id } : null
          const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')

          if (info) {info.stored_session_id = r.session_key || info.stored_session_id}

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
          cancelResumeScrollRef.current?.()
          cancelResumeScrollRef.current = scheduleResumeScrollToBottom(scrollRef)
          adoptAttachment(r, previousSid, previousSubscription)
        })
        .catch((e: Error) => {
          if (flight !== attachmentFlight.current) {return}
          sys(`error: ${e.message}`)
          patchUiState({ status: 'ready' })
        })
        .finally(() => finishAttachment(flight))
    },
    [adoptAttachment, discardStaleAttachment, finishAttachment, gw, resetSession, scrollRef, setHistoryItems, setSessionStartedAt, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      const flight = ++attachmentFlight.current
      pendingAttachments.current.add(flight)
      const current = captureDestination()
      const previousSid = current.sid
      const previousSubscription = previousSid ? canonicalSubscriptions.current.get(previousSid) : undefined

      const destination = current.sid === id || current.storedSid === id
        ? current : { ...current, sid: id, storedSid: id }

      patchOverlayState({ sessions: false })
      patchUiState({ status: 'resuming…' })

      ;(gw.isCanonical ? Promise.resolve(null) : rpc<SetupStatusResponse>('setup.status', {})).then(setup => {
        if (flight !== attachmentFlight.current) {return}

        if (setup?.provider_configured === false) {
          panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
          patchUiState({ status: 'setup required' })

          return
        }

        const pendingDetach = canonicalDetachFlights.current.get(id)
        const request = pendingDetach
          ? pendingDetach.then(() => flight === attachmentFlight.current
            ? gw.request<SessionResumeResponse>('session.resume', { cols: colsRef.current, session_id: id }) : null)
          : gw.request<SessionResumeResponse>('session.resume', { cols: colsRef.current, session_id: id })

        return request
          .then(raw => {
            const r = asRpcResult<SessionResumeResponse>(raw)

            if (flight !== attachmentFlight.current) {
              discardStaleAttachment(r)

              return
            }

            if (!r) {
              sys('error: invalid response: session.resume')

              return patchUiState({ status: 'ready' })
            }

            const info = r.info ? { ...r.info, stored_session_id: r.stored_session_id || r.info.stored_session_id } : null
            const running = Boolean(r.running || r.status === 'working' || r.status === 'waiting')

            if (info) {info.stored_session_id = r.session_key || info.stored_session_id || r.resumed}

            // A successful resume authorizes the requested source → canonical
            // successor mapping; ordinary focus changes never migrate input.
            migratePendingInputs(destination, r.session_id, info?.stored_session_id)
            resetSession()
            setSessionStartedAt(r.started_at ? r.started_at * 1000 : Date.now())

            const resumed = [...toTranscriptMessages(r.messages), ...liveSessionInflightMessages(r.inflight)]

            setHistoryItems(info ? [introMsg(info), ...resumed] : resumed)
            writeActiveSessionFile(r.resumed ?? r.session_id)
            patchUiState({
              busy: running,
              info,
              sid: r.session_id,
              gatewayConnected: true,
              status: statusFromLiveSession(r.status, running),
              usage: usageFrom(info)
            })
            gw.hydrateSharedPrompts?.(r)
            hydrateLiveSessionInflight(r.inflight)
            cancelResumeScrollRef.current?.()
            cancelResumeScrollRef.current = scheduleResumeScrollToBottom(scrollRef)

            if (previousSid && previousSid !== r.session_id) {
              void closeSession(previousSid)
            }

            adoptAttachment(r, previousSid, previousSubscription)
          })
          .catch((e: Error) => {
            if (flight !== attachmentFlight.current) {return}
            sys(`error: ${e.message}`)
            patchUiState({ status: 'ready' })
          })
      }).finally(() => finishAttachment(flight))
    },
    [adoptAttachment, closeSession, colsRef, discardStaleAttachment, finishAttachment, gw, panel, resetSession, rpc, scrollRef,
      setHistoryItems, setSessionStartedAt, sys]
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
      resumeById
    ]
  )
}
