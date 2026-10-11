/**
 * ChatSidebar — structured-events panel that sits next to the xterm.js
 * terminal in the dashboard Chat tab.
 *
 * Two WebSockets, one per concern:
 *
 *   1. **JSON-RPC sidecar** (`GatewayClient` → /api/ws) — a lightweight
 *      session used only for connection state (the "live" badge) and
 *      credential warnings. Independent of the PTY pane's session by
 *      design. The model badge does NOT come from here: it prefers the
 *      PTY chat session's runtime identity (`session.info` over the events
 *      feed — the model actually answering, which changes when a provider
 *      fallback replaces the configured primary mid-turn, #54509), falls
 *      back to the effective config model over REST (`/api/model/info`),
 *      and the model picker writes config over REST (`/api/model/set`)
 *      then offers a dashboard reload so the running chat adopts it.
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
  isEventsAuthRejectionMessage,
  isEventsFeedMessage,
  shouldRetryEventsClose
} from '@/lib/events-reconnect'
import { credentialWarning, sidecarErrorMessage } from '@/lib/chat-sidebar-banner'
import { titleFromSessionInfoPayload } from '@/lib/chat-title'

import { cn } from '@/lib/utils'
import { AlertCircle, ChevronDown, KeyRound, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

interface SessionInfo {
  cwd?: string
  model?: string
  provider?: string
  credential_warning?: string
  title?: string
}

// Auto-redial budget for the JSON-RPC sidecar (#95951). After this many
// bounded-backoff attempts the manual Reconnect affordance stays the only
// path, mirroring the events feed's give-up contract.
const SIDE_CAR_MAX_RECONNECT_ATTEMPTS = 5;

// A socket that opens and dies within this window is a flap, not a recovery:
// only a connection that stays open this long refills a reconnect budget.
// Shared by the JSON-RPC sidecar and the events feed (#129393).
const HEALTHY_OPEN_GRACE_MS = 10_000;

// Surfaced once when the redial budget is exhausted. Only this module may
// clear it (on the next successful open), matching how the events feed
// owns its own banner messages.
const SIDE_CAR_GAVE_UP_MESSAGE =
  "gateway sidecar disconnected — gave up after " +
  `${SIDE_CAR_MAX_RECONNECT_ATTEMPTS} attempts, use Reconnect`;

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

/** The runtime model a `session.info` payload reports, or undefined when it
 *  carries none (title-only updates) — config stays the badge's source then.
 *  `session.info` is surface-specific on the wire, so narrow defensively. */
function sessionInfoModel(payload: unknown): string | undefined {
  if (!payload || typeof payload !== 'object') return undefined
  const model = (payload as { model?: unknown }).model
  return typeof model === 'string' && model.trim() ? model : undefined
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
  const navigate = useNavigate()
  // `version` bumps on reconnect (manual button, profile/channel switch) and
  // re-runs the socket effects. The clients themselves live for the whole
  // component: the shared client keeps per-session seq watermarks and asks
  // the gateway to replay the gap on the next `connect()`, which only works
  // when the SAME instance survives the drop.
  const [version, setVersion] = useState(0)
  const gw = useMemo(() => new GatewayClient(), [])
  const feed = useMemo(() => new EventsFeedClient(), [])
  // Sidecar auto-redial budget (#95951). A ref, NOT effect state: the counter
  // must survive the [gw, version] effect re-runs a redial triggers, or the
  // budget resets every attempt and never exhausts.
  // Reset on a successful open and on scope switches.
  const sidecarRedialAttemptRef = useRef(0)
  const sidecarGaveUpRef = useRef(false)

  const [state, setState] = useState<ConnectionState>('idle')
  const [info, setInfo] = useState<SessionInfo>({})
  const [modelOpen, setModelOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Runtime model identity of the PTY chat session, from its `session.info`
  // broadcasts over the events feed. This is the model actually answering —
  // when a provider fallback replaces the configured primary mid-turn
  // (#54509), the end-of-turn `session.info` reports the fallback model, and
  // the badge must follow it rather than attribute the response to the
  // configured model that failed.
  const [runtimeModel, setRuntimeModel] = useState('')
  // The config fallback for the badge: config.yaml's main model
  // (`model.default`) via `/api/model/info` — the same value the Models page
  // writes and a new chat session boots from, shown until the PTY session
  // has broadcast a runtime identity. We deliberately don't use the
  // SIDECAR's `session.info` model: that's a one-time snapshot of the
  // throwaway sidecar agent taken when its session is created, and it never
  // updates when the model is changed elsewhere, so the badge would go
  // stale. Pass the chat profile explicitly so this card stays scoped to
  // the PTY even if the global dashboard switcher changes while the chat
  // is open.
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
    // Fresh PTY child on the new scope: its runtime identity is unknown until
    // its first `session.info` broadcast, so drop the previous chat's model.
    setRuntimeModel('')
    // Fresh scope, fresh sidecar redial budget (#95951).
    sidecarRedialAttemptRef.current = 0
    sidecarGaveUpRef.current = false
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
        console.warn(`[chat-sidebar] sidecar error: ${message}`)
        setError(sidecarErrorMessage(message))
      }
    })

    // Auto-redial after a transient drop (#95951): a dashboard service
    // restart closes the sidecar's WebSocket with 1012, and GatewayClient
    // deliberately delegates reconnect policy to this connection owner.
    // Bounded exponential backoff — the same shape the PTY pane uses —
    // capped at SIDE_CAR_MAX_RECONNECT_ATTEMPTS; after that the manual
    // Reconnect affordance stays the only path. A successful open resets
    // the counter; unmount or a scope switch (version bump) cancels the
    // pending timer because this effect tears down with the old client.
    let redialTimer: ReturnType<typeof setTimeout> | null = null;
    let healthyOpenTimer: ReturnType<typeof setTimeout> | null = null;
    // onState replays the current state synchronously. Ignore only that
    // subscription-time snapshot; real transitions in the same effect must
    // still consume the retry budget.
    let replayingInitialState = true;
    queueMicrotask(() => {
      replayingInitialState = false;
    });
    const offRedial = gw.onState((s) => {
      if (replayingInitialState) {
        return;
      }
      if (s === "open") {
        // A pending redialTimer would bump the version and tear down the
        // connection that just opened (#129393).
        if (redialTimer) {
          clearTimeout(redialTimer);
          redialTimer = null;
        }
        if (healthyOpenTimer) {
          clearTimeout(healthyOpenTimer);
        }
        // Do not reset the budget on every open: an open→immediate-close
        // cycle would otherwise reset it forever. Reset only after a stable
        // connection has remained open for the grace period.
        healthyOpenTimer = setTimeout(() => {
          healthyOpenTimer = null;
          sidecarRedialAttemptRef.current = 0;
          if (sidecarGaveUpRef.current) {
            sidecarGaveUpRef.current = false;
            setError((current: string | null) =>
              current === SIDE_CAR_GAVE_UP_MESSAGE ? null : current,
            );
          }
        }, HEALTHY_OPEN_GRACE_MS);
        return;
      }
      if (s !== "closed" && s !== "error") {
        return;
      }
      if (healthyOpenTimer) {
        clearTimeout(healthyOpenTimer);
        healthyOpenTimer = null;
      }
      if (cancelled || redialTimer) {
        return;
      }
      // The attempt counter lives in a ref: each redial rebuilds the client
      // and re-runs this effect, so a closure-local counter would reset and
      // the budget would never exhaust (#95951).
      if (sidecarRedialAttemptRef.current >= SIDE_CAR_MAX_RECONNECT_ATTEMPTS) {
        // Mirror the events feed's give-up contract: say so once, then the
        // manual Reconnect affordance stays the only path. Cleared again if
        // a later connection does open (manual reconnect followed by a
        // within-budget drop).
        if (!sidecarGaveUpRef.current) {
          sidecarGaveUpRef.current = true;
          setError((current) => current ?? SIDE_CAR_GAVE_UP_MESSAGE);
        }
        return;
      }
      const attempt = sidecarRedialAttemptRef.current;
      sidecarRedialAttemptRef.current += 1;
      const delayMs = Math.min(250 * 2 ** attempt, 3000);
      redialTimer = setTimeout(() => {
        redialTimer = null;
        if (!cancelled) {
          setVersion((v) => v + 1);
        }
      }, delayMs);
    });

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
          console.warn(`[chat-sidebar] sidecar connect failed: ${e.message}`)
          setError(sidecarErrorMessage(e.message))
        }
      })

    return () => {
      cancelled = true
      if (redialTimer) {
        clearTimeout(redialTimer)
        redialTimer = null
      }
      if (healthyOpenTimer) {
        clearTimeout(healthyOpenTimer)
        healthyOpenTimer = null
      }
      offRedial()
      offState()
      offSessionInfo()
      offError()
      gw.close()
    }
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
    let healthyOpenTimer: ReturnType<typeof setTimeout> | null = null
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
      if (healthyOpenTimer) {
        clearTimeout(healthyOpenTimer)
        healthyOpenTimer = null
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
      console.warn(`[chat-sidebar] events feed closed code=${code ?? 'none'}`)
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
        clearEventsBanner()
        // Same rule as the sidecar: an open that dies within the grace window
        // is a flap, not a recovery, so it must not refill the ladder.
        if (healthyOpenTimer) {
          clearTimeout(healthyOpenTimer)
        }
        healthyOpenTimer = setTimeout(() => {
          healthyOpenTimer = null
          attempt = 0
        }, HEALTHY_OPEN_GRACE_MS)
      }
    })

    const offSessionInfo = feed.on('session.info', ev => {
      const title = titleFromSessionInfoPayload(ev.payload)
      if (title !== undefined) {
        onSessionTitleChange?.(title)
      }
      // Runtime identity of the PTY chat session — the model actually
      // answering, which a provider fallback can swap mid-turn (#54509).
      // The gateway re-emits `session.info` at every turn end (and on model
      // switches), so this stays current without any new plumbing.
      const model = sessionInfoModel(ev.payload)
      if (model !== undefined) {
        setRuntimeModel(model)
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
      if (healthyOpenTimer) {
        clearTimeout(healthyOpenTimer)
        healthyOpenTimer = null
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

  // Runtime-first (#54509): the PTY session's `session.info` reports the
  // model that is actually answering, so a provider-fallback swap shows the
  // fallback model, not the configured primary that failed. Config remains
  // the fallback until the PTY has broadcast a runtime identity (fresh chat,
  // events feed still connecting).
  const modelName = runtimeModel || effectiveModel || info.model || '—'
  const modelLabel = modelName.split('/').slice(-1)[0] ?? '—'
  const credential = credentialWarning(info.credential_warning)
  const banner = error ?? credential?.message ?? null
  const showReload = isEventsAuthRejectionMessage(error)

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

            {error && showReload && (
              <Button
                size="sm"
                outlined
                className="mt-1"
                onClick={() => window.location.reload()}
                prefix={<RefreshCw />}
              >
                Reload page
              </Button>
            )}
            {error && !showReload && (
              <Button size="sm" outlined className="mt-1" onClick={reconnect} prefix={<RefreshCw />}>
                Reconnect side panel
              </Button>
            )}
            {!error && credential && (
              <div className="mt-1 flex flex-wrap gap-2">
                <Button
                  size="sm"
                  outlined
                  prefix={<KeyRound />}
                  // Router navigation: a full page load would tear down the
                  // xterm scrollback and the chat sockets. (The mobile portal
                  // still lives under ChatPage, so router context is present.)
                  onClick={() => navigate('/env')}
                >
                  Add key
                </Button>
                <Button size="sm" outlined onClick={() => setModelOpen(true)}>
                  Switch model
                </Button>
              </div>
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
