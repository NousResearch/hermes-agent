/**
 * ChatSidebar — structured-events panel that sits next to the xterm.js
 * terminal in the dashboard Chat tab.
 *
 * Two WebSockets, one per concern:
 *
 *   1. **JSON-RPC sidecar** (`GatewayClient` → /api/ws) — a lightweight
 *      session used only for connection state (the "live" badge) and
 *      credential warnings. Independent of the PTY pane's session by
 *      design. The model badge does NOT come from here: it reads the
 *      effective config model over REST (`/api/model/info`), and the model
 *      picker writes config over REST (`/api/model/set`) then offers a
 *      dashboard reload so the running chat adopts the new model.
 *
 *   2. **Event subscriber** (/api/events?channel=…) — passive, receives
 *      every dispatcher emit from the PTY-side `tui_gateway.entry` that
 *      the dashboard fanned out.  The sidebar uses it for `session.info`
 *      (live chat title) and `dashboard.new_session_requested`.  The
 *      `channel` id ties this listener to the same chat tab's PTY child —
 *      see `ChatPage.tsx` for where the id is generated.  Transient drops
 *      (gateway restart, network blip) auto-reconnect with exponential
 *      backoff; auth rejections are terminal.  See `lib/events-reconnect`.
 *
 * Best-effort throughout: WS failures show in the badge / banner, the
 * terminal pane keeps working unimpaired.
 */

import { Button } from '@nous-research/ui/ui/components/button'
import { Badge } from '@nous-research/ui/ui/components/badge'
import { Card } from '@nous-research/ui/ui/components/card'

import { ModelPickerDialog } from '@/components/ModelPickerDialog'
import { ModelReloadConfirm } from '@/components/ModelReloadConfirm'
import { ReasoningPicker } from '@/components/ReasoningPicker'
import { GatewayClient, type ConnectionState } from '@/lib/gatewayClient'
import { EventsFeedClient } from '@/lib/eventsFeedClient'
import { api } from '@/lib/api'
import {
  EVENTS_MAX_RECONNECT_ATTEMPTS,
  eventsGaveUpMessage,
  eventsReconnectDelayMs,
  eventsReconnectingMessage,
  eventsRejectedMessage,
  isEventsAuthRejection,
  isEventsFeedMessage,
  shouldRetryEventsClose
} from '@/lib/events-reconnect'
import { titleFromSessionInfoPayload } from '@/lib/chat-title'

import { cn } from '@/lib/utils'
import { AlertCircle, ChevronDown, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

interface SessionInfo {
  cwd?: string
  model?: string
  provider?: string
  credential_warning?: string
  title?: string
}

const STATE_LABEL: Record<ConnectionState, string> = {
  idle: 'idle',
  connecting: 'connecting',
  open: 'live',
  closed: 'closed',
  error: 'error'
}

const STATE_TONE: Record<ConnectionState, 'secondary' | 'warning' | 'success' | 'destructive'> = {
  idle: 'secondary',
  connecting: 'warning',
  open: 'success',
  closed: 'secondary',
  error: 'destructive'
}

interface ChatSidebarProps {
  channel: string
  /** Chat profile from the dashboard switcher / URL scope. */
  profile?: string
  className?: string
  onDashboardNewSessionRequest?: () => void
  onSessionTitleChange?: (title: string | null) => void
}

/** Build the ``session.create`` params for the sidecar session.
 *
 * Extracted from the effect below so the invariant — close_on_disconnect
 * is set, source is "tool", and the profile is forwarded when present —
 * can be tested without reading component source text. See
 * ``chat-sidebar-session-params.test.ts``.
 */
export function sidecarSessionCreateParams(profile?: string): Record<string, unknown> {
  return {
    close_on_disconnect: true,
    source: 'tool',
    ...(profile ? { profile } : {})
  }
}

export function ChatSidebar({
  channel,
  profile,
  className,
  onDashboardNewSessionRequest,
  onSessionTitleChange
}: ChatSidebarProps) {
  // `version` bumps on reconnect (manual button, profile/channel switch) and
  // re-runs the socket effects. The clients themselves live for the whole
  // component: the shared client keeps per-session seq watermarks and asks
  // the gateway to replay the gap on the next `connect()`, which only works
  // when the SAME instance survives the drop.
  const [version, setVersion] = useState(0)
  const gw = useMemo(() => new GatewayClient(), [])
  const feed = useMemo(() => new EventsFeedClient(), [])

  const [state, setState] = useState<ConnectionState>('idle')
  const [info, setInfo] = useState<SessionInfo>({})
  const [modelOpen, setModelOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // The badge shows config.yaml's main model (`model.default`) via
  // `/api/model/info` — the same value the Models page writes and a new chat
  // session boots from. We deliberately don't use the sidecar's `session.info`
  // model: that's a one-time snapshot of the throwaway sidecar agent taken when
  // its session is created, and it never updates when the model is changed
  // elsewhere, so the badge would go stale. Pass the chat profile explicitly so
  // this card stays scoped to the PTY even if the global dashboard switcher
  // changes while the chat is open.
  const [effectiveModel, setEffectiveModel] = useState('')
  // Whether the effective model supports reasoning effort — gates the
  // ReasoningPicker. Read from the same `/api/model/info` capabilities the
  // (currently unused) ModelInfoCard surfaces, so the dashboard exposes a
  // control to *set* the level, not just a read-only "Reasoning" badge.
  const [supportsReasoning, setSupportsReasoning] = useState(false)
  // Bumped on model change/save so ReasoningPicker re-reads the saved effort
  // (config is profile-scoped the same way the model badge is).
  const [modelRefreshKey, setModelRefreshKey] = useState(0)
  // Set after the picker saves a model and the user declines the reload: config
  // is updated but the running session keeps its model until rebuilt.
  const [modelNotice, setModelNotice] = useState<string | null>(null)
  // Short name of a just-saved model awaiting confirm to reload (a fresh chat
  // session is how the running chat adopts it; we confirm before discarding it).
  const [pendingReloadModel, setPendingReloadModel] = useState<string | null>(null)

  const refreshEffectiveModel = useCallback(() => {
    void api
      .getModelInfo(profile)
      .then(r => {
        if (r?.model) setEffectiveModel(String(r.model))
        setSupportsReasoning(!!r?.capabilities?.supports_reasoning)
        // Bump so ReasoningPicker re-reads the saved effort for the new model.
        setModelRefreshKey(k => k + 1)
      })
      .catch(() => {
        // Best-effort: keep the last known label rather than blanking it.
      })
  }, [profile])

  // Profile or PTY channel change tears down both WebSockets. Bump `version`
  // (same path as the manual Reconnect button) so the gateway client is
  // recreated and the events feed resubscribes — otherwise the old events
  // socket's close handler can leave a stale error banner after a switch.
  const scopeKey = `${channel}\0${profile ?? ''}`
  const prevScopeKey = useRef<string | null>(null)
  useEffect(() => {
    if (prevScopeKey.current === null) {
      prevScopeKey.current = scopeKey
      return
    }
    if (prevScopeKey.current === scopeKey) return
    prevScopeKey.current = scopeKey
    setError(null)
    setVersion(v => v + 1)
  }, [scopeKey])

  useEffect(() => {
    let cancelled = false
    queueMicrotask(() => {
      if (cancelled) return
      setInfo({})
      setError(null)
    })
    const offState = gw.onState(setState)

    const offSessionInfo = gw.on('session.info', ev => {
      // session.info is surface-specific on the wire; narrow to the fields this sidebar reads.
      const payload = ev.payload as SessionInfo | undefined

      if (payload) {
        setInfo(prev => ({ ...prev, ...payload }))
      }
    })

    const offError = gw.on('error', ev => {
      const message = ev.payload?.message

      if (message) {
        setError(message)
      }
    })

    // Create the sidecar session so the gateway surfaces session-scoped
    // signals (connection state, credential warnings). It's independent of the
    // PTY pane's session by design. The model picker no longer rides this
    // session — it writes config.yaml over REST — so we don't track its id.
    gw.connect()
      .then(() => {
        if (cancelled) {
          return
        }
        // close_on_disconnect: the gateway reaps this sidecar session (and its
        // slash_worker subprocess) when the WS drops, instead of leaking it.
        return gw.request<{ session_id: string }>('session.create', sidecarSessionCreateParams(profile))
      })
      .catch((e: Error) => {
        if (!cancelled) {
          setError(e.message)
        }
      })

    return () => {
      cancelled = true
      offState()
      offSessionInfo()
      offError()
      gw.close()
    }
    // `profile` is read from render; scope changes bump `version` → redial.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gw, version])

  // Event subscriber WebSocket — receives the rebroadcast of every
  // dispatcher emit from the PTY child's gateway.  See /api/pub +
  // /api/events in hermes_cli/web_server.py for the broadcast hop.
  //
  // Framing, dispatch and connect timeout come from the shared JSON-RPC
  // client (`EventsFeedClient`); this effect owns only the retry policy and
  // the banner. Failures (auth/loopback rejection, server too old to expose
  // the endpoint, transient drops) surface in the same banner as the
  // JSON-RPC sidecar so the sidebar matches its documented best-effort UX
  // and the user always has a reconnect affordance.
  useEffect(() => {
    if (!channel) {
      return
    }
    let unmounting = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let attempt = 0

    // The banner is shared with `info.credential_warning` and the JSON-RPC
    // sidecar, and `error` is those messages' only home — the sidecar does
    // not re-emit. So the events feed may only write over an empty banner
    // or one of its own messages, and may only clear its own.
    const surface = (msg: string) =>
      !unmounting && setError(current => (isEventsFeedMessage(current) ? msg : (current ?? msg)))

    const clearEventsBanner = () => !unmounting && setError(current => (isEventsFeedMessage(current) ? null : current))

    // Single scheduling path: the client collapses `error` + `close` of one
    // socket generation into one `closed` transition, so only one timer is
    // ever queued per drop.
    const scheduleReconnect = () => {
      if (unmounting || reconnectTimer) {
        return
      }
      if (attempt >= EVENTS_MAX_RECONNECT_ATTEMPTS) {
        surface(eventsGaveUpMessage())
        return
      }

      const delay = eventsReconnectDelayMs(attempt)
      attempt += 1
      surface(eventsReconnectingMessage(delay))

      reconnectTimer = setTimeout(() => {
        reconnectTimer = null
        void connect()
      }, delay)
    }

    const connect = async () => {
      if (unmounting) {
        return
      }
      try {
        // Re-minted every attempt: tickets are single-use with a short TTL,
        // so a reconnect cannot replay the URL from the first connection.
        await feed.connect(channel)
      } catch {
        // Connect-phase failures (ticket mint, handshake timeout, close
        // during handshake) reach the reconnect ladder through the close
        // handler when a socket existed; a pre-socket failure has no close
        // event, so schedule here. `onClose` de-dupes via `reconnectTimer`.
        if (!unmounting && feed.lastCloseCode === null) {
          scheduleReconnect()
        }
      }
    }

    const offClose = feed.onClose(code => {
      if (unmounting) {
        return
      }
      if (code !== undefined && isEventsAuthRejection(code)) {
        surface(eventsRejectedMessage(code))
        return
      }
      // `undefined` = handshake timeout / error without a close frame.
      if (shouldRetryEventsClose(code)) {
        scheduleReconnect()
      }
    })

    const offState = feed.onState(state => {
      if (state === 'open') {
        attempt = 0
        clearEventsBanner()
      }
    })

    const offSessionInfo = feed.on('session.info', ev => {
      const title = titleFromSessionInfoPayload(ev.payload)
      if (title !== undefined) {
        onSessionTitleChange?.(title)
      }
    })
    const offNewSession = feed.on('dashboard.new_session_requested', () => {
      onDashboardNewSessionRequest?.()
    })

    void connect()

    return () => {
      unmounting = true
      if (reconnectTimer) {
        clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
      offClose()
      offState()
      offSessionInfo()
      offNewSession()
      feed.close()
    }
  }, [channel, feed, onDashboardNewSessionRequest, onSessionTitleChange, version])

  // Seed the badge on mount and re-read it whenever the sockets are rebuilt
  // (a profile/channel switch bumps `version`).
  useEffect(() => {
    refreshEffectiveModel()
  }, [refreshEffectiveModel, version])

  const reconnect = useCallback(() => {
    setError(null)
    setModelNotice(null)
    setPendingReloadModel(null)
    setVersion(v => v + 1)
  }, [])

  // The picker writes config.yaml over REST and reloads — it doesn't ride the
  // sidecar gateway session, so it's available whenever the sidebar is mounted.
  const modelName = effectiveModel || info.model || '—'
  const modelLabel = modelName.split('/').slice(-1)[0] ?? '—'
  const banner = error ?? info.credential_warning ?? null

  return (
    <aside
      className={cn(
        'flex h-full w-full min-w-0 shrink-0 flex-col gap-3 overflow-y-auto overflow-x-hidden pr-1',
        className
      )}
    >
      <Card className="flex items-center justify-between gap-2 px-3 py-2">
        <div className="min-w-0 flex-1">
          <div className="text-display text-xs tracking-wider text-text-tertiary">model</div>

          <Button
            ghost
            size="sm"
            onClick={() => setModelOpen(true)}
            className={cn(
              'max-w-full min-w-0 px-0 py-0',
              'self-start normal-case tracking-normal text-sm font-medium',
              'hover:underline disabled:no-underline'
            )}
            title={modelName === '—' ? 'switch model' : modelName}
          >
            <span className="flex min-w-0 max-w-full items-center gap-1">
              <span className="truncate">{modelLabel}</span>

              <ChevronDown className="size-3.5 shrink-0 text-text-secondary" />
            </span>
          </Button>
        </div>

        <Badge tone={STATE_TONE[state]} className="shrink-0">
          {STATE_LABEL[state]}
        </Badge>
      </Card>

      {supportsReasoning && (
        <Card className="py-0">
          <ReasoningPicker
            currentModel={modelName}
            profile={profile}
            refreshKey={modelRefreshKey}
            onChanged={effort =>
              setModelNotice(
                `Reasoning effort set to ${effort}. Run /new or refresh the page to apply it to this chat.`
              )
            }
          />
        </Card>
      )}

      {modelNotice && (
        <Card className="flex items-start gap-2 border-warning/40 bg-warning/5 px-3 py-2 text-xs">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" />

          <div className="wrap-break-word min-w-0 flex-1 text-text-secondary">{modelNotice}</div>
        </Card>
      )}

      {banner && (
        <Card className="flex items-start gap-2 border-destructive/40 bg-destructive/5 px-3 py-2 text-xs">
          <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-destructive" />

          <div className="min-w-0 flex-1">
            <div className="wrap-break-word text-destructive">{banner}</div>

            {error && (
              <Button size="sm" outlined className="mt-1" onClick={reconnect} prefix={<RefreshCw />}>
                reconnect events feed
              </Button>
            )}
          </div>
        </Card>
      )}

      {modelOpen && (
        <ModelPickerDialog
          // Same path the Models page uses (REST /api/model/set), not the
          // sidecar config.set RPC, which didn't reliably land in the
          // config.yaml the agent boots from. Always persisted (alwaysGlobal).
          loader={() => api.getModelOptions(profile)}
          alwaysGlobal
          onApply={async ({ provider, model, confirmExpensiveModel }) => {
            setModelNotice(null)
            setPendingReloadModel(null)
            const result = await api.setModelAssignment(
              {
                confirm_expensive_model: confirmExpensiveModel,
                scope: 'main',
                provider,
                model
              },
              profile
            )
            // confirm_required => the dialog shows the expensive-model prompt
            // and calls back; don't announce until the user confirms.
            if (!result.confirm_required) {
              refreshEffectiveModel()
              // Ask before reloading: applying the model starts a fresh chat.
              setPendingReloadModel(model.split('/').slice(-1)[0])
            }
            return result
          }}
          onClose={() => {
            setModelOpen(false)
            refreshEffectiveModel()
          }}
        />
      )}

      <ModelReloadConfirm
        model={pendingReloadModel}
        onCancel={() => {
          const m = pendingReloadModel
          setPendingReloadModel(null)
          setModelNotice(`Model set to ${m}. Run /new or refresh the page to apply it to this chat.`)
        }}
      />
    </aside>
  )
}
