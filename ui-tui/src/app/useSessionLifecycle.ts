import { writeFileSync } from 'node:fs'

import type { ScrollBoxHandle } from '@hermes/ink'
import { evictInkCaches } from '@hermes/ink'
import { useCallback, type RefObject } from 'react'

import { buildSetupRequiredSections, setupRequiredTitle } from '../content/setup.js'
import { translate, type TranslationKey } from '../i18n/index.js'
import { introMsg, toTranscriptMessages } from '../domain/messages.js'
import { ZERO } from '../domain/usage.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  SessionCloseResponse,
  SessionCreateResponse,
  SessionResumeResponse,
  SessionTitleResponse,
  SetupStatusResponse
} from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import type { Msg, PanelSection, SessionInfo, Usage } from '../types.js'

import type { ComposerActions, GatewayRpc, StateSetter } from './interfaces.js'
import { patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const usageFrom = (info: null | SessionInfo): Usage => (info?.usage ? { ...ZERO, ...info.usage } : ZERO)

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

  const ti = (key: TranslationKey, vars?: Record<string, string | number>) => translate(getUiState().locale, key, vars)

  const resetSession = useCallback(() => {
    turnController.fullReset()
    setVoiceRecording(false)
    setVoiceProcessing(false)
    patchUiState({ bgTasks: new Set(), info: null, sid: null, usage: ZERO })
    setHistoryItems([])
    setLastUserMsg('')
    setStickyPrompt('')
    composerActions.setPasteSnips([])
    // Half-prune: new session has new keys, but keep a warm pool in case
    // the user resumes back to the prior session.
    evictInkCaches('half')
  }, [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt, setVoiceProcessing, setVoiceRecording])

  const resetVisibleHistory = useCallback(
    (info: null | SessionInfo = null) => {
      turnController.idle()
      turnController.clearReasoning()
      turnController.turnTools = []
      turnController.persistedToolLabels.clear()

      setHistoryItems(info ? [introMsg(info)] : [])
      setStickyPrompt('')
      setLastUserMsg('')
      composerActions.setPasteSnips([])
      patchTurnState({ activity: [] })
      patchUiState({ info, usage: usageFrom(info) })
    },
    [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt]
  )

  const newSession = useCallback(
    async (msg?: string, title?: string) => {
      const setup = await rpc<SetupStatusResponse>('setup.status', {})

      if (setup?.provider_configured === false) {
        const { locale } = getUiState()
        panel(setupRequiredTitle(locale), buildSetupRequiredSections(locale))
        patchUiState({ status: 'setup required' })

        return
      }

      await closeSession(getUiState().sid)

      const r = await rpc<SessionCreateResponse>('session.create', { cols: colsRef.current })

      if (!r) {
        return patchUiState({ status: 'ready' })
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
        sys(ti('transcript.credentialWarning', { message: info.credential_warning }))
      }

      if (info?.config_warning) {
        sys(ti('transcript.configWarning', { message: info.config_warning }))
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
            sys(ti('session.titleSet', { title: nextTitle, suffix }))
          })
          .catch((err: unknown) => {
            if (getUiState().sid !== r.session_id) {
              return
            }

            const message = err instanceof Error ? err.message : String(err)
            sys(ti('session.titleSetFailed', { message }))
          })
      }
    },
    [closeSession, colsRef, panel, resetSession, rpc, setHistoryItems, setSessionStartedAt, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      patchOverlayState({ picker: false })
      patchUiState({ status: 'resuming…' })

      rpc<SetupStatusResponse>('setup.status', {}).then(setup => {
        if (setup?.provider_configured === false) {
          const { locale } = getUiState()
          panel(setupRequiredTitle(locale), buildSetupRequiredSections(locale))
          patchUiState({ status: 'setup required' })

          return
        }

        closeSession(getUiState().sid === id ? null : getUiState().sid).then(() =>
          gw
            .request<SessionResumeResponse>('session.resume', { cols: colsRef.current, session_id: id })
            .then(raw => {
              const r = asRpcResult<SessionResumeResponse>(raw)

              if (!r) {
                sys(ti('errors.invalidResponse', { method: 'session.resume' }))

                return patchUiState({ status: 'ready' })
              }

              resetSession()
              setSessionStartedAt(Date.now())

              const resumed = toTranscriptMessages(r.messages)

              setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
              writeActiveSessionFile(r.resumed ?? r.session_id)
              patchUiState({
                info: r.info ?? null,
                sid: r.session_id,
                status: 'ready',
                usage: usageFrom(r.info ?? null)
              })
              setTimeout(() => scrollRef.current?.scrollToBottom(), 0)
            })
            .catch((e: Error) => {
              sys(ti('errors.rpc', { message: e.message }))
              patchUiState({ status: 'ready' })
            })
        )
      })
    },
    [closeSession, colsRef, gw, panel, resetSession, rpc, scrollRef, setHistoryItems, setSessionStartedAt, sys]
  )

  const guardBusySessionSwitch = useCallback(
    (what = 'switch sessions') => {
      if (!getUiState().busy) {
        return false
      }

      sys(ti('session.switchBusy', { what }))

      return true
    },
    [sys]
  )

  return {
    closeSession,
    guardBusySessionSwitch,
    newSession,
    resetSession,
    resetVisibleHistory,
    resumeById,
    trimLastExchange: trimTail
  }
}
