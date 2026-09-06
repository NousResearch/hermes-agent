import { useEffect, useMemo, useState } from 'react'
import { Bot, Check, ExternalLink, QrCode, Save, X } from 'lucide-react'
import * as QRCode from 'qrcode'
import { Badge } from '@nous-research/ui/ui/components/badge'
import { Button } from '@nous-research/ui/ui/components/button'
import { Input } from '@nous-research/ui/ui/components/input'
import { Spinner } from '@nous-research/ui/ui/components/spinner'
import { api, getManagementProfile, HERMES_BASE_PATH } from '@/lib/api'
import type { MessagingPlatform, TelegramOnboardingStartResponse } from '@/lib/api'
import { readTelegramSetup, saveTelegramSetup, isTerminalTelegramOnboardingError } from '@/lib/telegram-onboarding'
const TELEGRAM_USER_ID_RE = /^\d+$/
function formatExpiry(expiresAt: string): string {
  const ms = Date.parse(expiresAt) - Date.now()
  if (!Number.isFinite(ms) || ms <= 0) return 'expired'
  const seconds = Math.ceil(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const rest = seconds % 60
  return `${minutes}:${rest.toString().padStart(2, '0')}`
}

export function TelegramOnboardingPanel({
  onManualSetup,
  onChanged,
  onRestartNeeded,
  platform,
  setRestartNeeded,
  showToast
}: {
  onManualSetup: () => void
  onChanged: () => Promise<void>
  onRestartNeeded: () => void
  platform: MessagingPlatform
  setRestartNeeded: (needed: boolean) => void
  showToast: (message: string, type: 'success' | 'error') => void
}) {
  const storageKey = `hermes:telegram-setup:${HERMES_BASE_PATH}:${getManagementProfile()}`
  const [setup, setSetup] = useState<TelegramOnboardingStartResponse | null>(() => readTelegramSetup(storageKey))
  const [qrDataUrl, setQrDataUrl] = useState('')
  const [phase, setPhase] = useState<'idle' | 'starting' | 'waiting' | 'ready' | 'applying'>(() =>
    readTelegramSetup(storageKey) ? 'waiting' : 'idle'
  )
  const [botUsername, setBotUsername] = useState<string | null>(null)
  const [allowedIds, setAllowedIds] = useState<string[]>([])
  const [detectedOwnerId, setDetectedOwnerId] = useState<string | null>(null)
  const [newAllowedId, setNewAllowedId] = useState('')
  const [error, setError] = useState('')
  const [tick, setTick] = useState(0)

  useEffect(() => {
    saveTelegramSetup(storageKey, setup)
  }, [setup, storageKey])

  useEffect(() => {
    if (!setup) return
    let cancelled = false
    void QRCode.toDataURL(setup.qr_payload, { margin: 1, width: 224 })
      .then(url => {
        if (!cancelled) setQrDataUrl(url)
      })
      .catch(() => {
        if (!cancelled) setError('QR code unavailable. Use Open Telegram to continue.')
      })
    return () => {
      cancelled = true
    }
  }, [setup])

  useEffect(() => {
    if (!setup || phase !== 'waiting') return
    let cancelled = false
    let timeout: ReturnType<typeof setTimeout> | null = null

    const poll = async () => {
      try {
        const status = await api.getTelegramOnboardingStatus(setup.pairing_id)
        if (cancelled) return
        if (status.status === 'ready') {
          if (status.saved) {
            setSetup(null)
            setQrDataUrl('')
            setPhase('idle')
            void onChanged()
            return
          }
          setSetup(current => (current ? { ...current, expires_at: status.expires_at } : null))
          setPhase('ready')
          setBotUsername(status.bot_username ?? null)
          setError('')
          if (status.owner_user_id && TELEGRAM_USER_ID_RE.test(status.owner_user_id)) {
            setDetectedOwnerId(status.owner_user_id)
            setAllowedIds([status.owner_user_id])
          }
          return
        }
        if (status.expires_at !== setup.expires_at) {
          setSetup(current => (current ? { ...current, expires_at: status.expires_at } : null))
        }
        setError('')
        timeout = setTimeout(poll, 2000)
      } catch (pollError) {
        if (cancelled) return

        // The Worker may have extended confirmation while this client was
        // offline. Only a terminal response can retire a waiting attempt.
        if (isTerminalTelegramOnboardingError(pollError)) {
          setSetup(null)
          setQrDataUrl('')
          setPhase('idle')
          setError('Telegram pairing expired. Start a new QR setup to try again.')
          return
        }

        setError(`Still waiting for Telegram. Retrying after: ${pollError}`)
        timeout = setTimeout(poll, 2000)
      }
    }

    timeout = setTimeout(poll, 1200)
    return () => {
      cancelled = true
      if (timeout) clearTimeout(timeout)
    }
  }, [phase, setup, onChanged])

  useEffect(() => {
    if (!setup) return
    const timer = setInterval(() => setTick(value => value + 1), 1000)
    return () => clearInterval(timer)
  }, [setup])

  useEffect(() => {
    if (!setup || phase !== 'ready') return
    const timer = setTimeout(
      () => {
        setSetup(null)
        setQrDataUrl('')
        setPhase('idle')
        setAllowedIds([])
        setDetectedOwnerId(null)
        setBotUsername(null)
        setError('Telegram confirmation expired. Start a new setup or connect the existing bot with Manual setup.')
      },
      Math.max(0, Date.parse(setup.expires_at) - Date.now())
    )
    return () => clearTimeout(timer)
  }, [phase, setup])

  const resetSetup = () => {
    setSetup(null)
    setQrDataUrl('')
    setPhase('idle')
    setBotUsername(null)
    setAllowedIds([])
    setDetectedOwnerId(null)
    setNewAllowedId('')
    setError('')
  }

  const start = async () => {
    setPhase('starting')
    setError('')
    setBotUsername(null)
    setAllowedIds([])
    setDetectedOwnerId(null)
    setNewAllowedId('')
    try {
      const res = await api.startTelegramOnboarding({ bot_name: 'Hermes Agent' })
      setSetup(res)
      setPhase('waiting')
    } catch (startError) {
      setPhase('idle')
      setError(String(startError))
    }
  }

  const cancel = async () => {
    if (setup) {
      try {
        await api.cancelTelegramOnboarding(setup.pairing_id)
      } catch (cancelError) {
        setError(`Could not cancel Telegram setup. Try Cancel again: ${cancelError}`)
        return
      }
    }
    resetSetup()
  }

  const addAllowedId = () => {
    const trimmed = newAllowedId.trim()
    if (!TELEGRAM_USER_ID_RE.test(trimmed)) {
      setError('Allowed Telegram user IDs must be numeric.')
      return
    }
    setError('')
    setAllowedIds(ids => (ids.includes(trimmed) ? ids : [...ids, trimmed]))
    setNewAllowedId('')
  }

  // restart_started only means the `hermes gateway restart` child spawned —
  // not that the restart will succeed (e.g. systemd linger missing, service
  // manager failure). Poll the action status briefly and surface a non-zero
  // exit via the manual-restart banner. Note: in no-service installs the
  // child becomes the foreground gateway and never exits, so "still running
  // when the window closes" counts as success.
  const watchRestartOutcome = async () => {
    for (let i = 0; i < 20; i++) {
      await new Promise(resolve => setTimeout(resolve, 1500))
      try {
        const st = await api.getActionStatus('gateway-restart', 5)
        if (st.running) continue
        if (st.exit_code !== 0 && st.exit_code !== null) {
          onRestartNeeded()
          showToast(`Gateway restart failed (exit ${st.exit_code}) — restart manually`, 'error')
        }
        return
      } catch {
        // transient fetch error; keep polling
      }
    }
  }

  const apply = async () => {
    if (!setup) return
    if (allowedIds.length === 0) {
      setError('Add at least one allowed Telegram user ID.')
      return
    }
    setPhase('applying')
    setError('')
    try {
      const result = await api.applyTelegramOnboarding(setup.pairing_id, {
        allowed_user_ids: allowedIds
      })
      resetSetup()
      if (result.restart_started) {
        showToast('Telegram saved; gateway restarting…', 'success')
        setRestartNeeded(false)
        setTimeout(() => void onChanged(), 4000)
        void watchRestartOutcome()
      } else if (result.restart_started === undefined && result.needs_restart) {
        try {
          await api.restartGateway()
          showToast('Telegram saved; gateway restarting…', 'success')
          setRestartNeeded(false)
          setTimeout(() => void onChanged(), 4000)
        } catch (restartError) {
          onRestartNeeded()
          showToast(`Telegram saved; gateway restart failed: ${restartError}`, 'error')
        }
      } else {
        onRestartNeeded()
        const detail = result.restart_error ? `: ${result.restart_error}` : ''
        showToast(`Telegram saved; gateway restart failed${detail}`, 'error')
      }
      await onChanged()
    } catch (applyError) {
      if (isTerminalTelegramOnboardingError(applyError)) {
        resetSetup()
        setError('Telegram setup is no longer available. Start a new setup or use Manual setup for an existing bot.')
      } else {
        setPhase('ready')
        setError(String(applyError))
      }
    }
  }

  const expiresIn = useMemo(
    () => (setup ? formatExpiry(setup.expires_at) : ''),
    // tick keeps the memo fresh without recalculating on every render branch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setup, tick]
  )

  return (
    <TelegramSetupView
      {...{
        platform,
        phase,
        start,
        onManualSetup,
        error,
        setup,
        qrDataUrl,
        botUsername,
        detectedOwnerId,
        allowedIds,
        setAllowedIds,
        newAllowedId,
        setNewAllowedId,
        addAllowedId,
        apply,
        cancel,
        expiresIn
      }}
    />
  )
}

interface TelegramSetupViewProps {
  platform: MessagingPlatform
  phase: 'idle' | 'starting' | 'waiting' | 'ready' | 'applying'
  start: () => Promise<void>
  onManualSetup: () => void
  error: string
  setup: TelegramOnboardingStartResponse | null
  qrDataUrl: string
  botUsername: string | null
  detectedOwnerId: string | null
  allowedIds: string[]
  setAllowedIds: React.Dispatch<React.SetStateAction<string[]>>
  newAllowedId: string
  setNewAllowedId: (value: string) => void
  addAllowedId: () => void
  apply: () => Promise<void>
  cancel: () => Promise<void>
  expiresIn: string
}

function TelegramSetupView({
  platform,
  phase,
  start,
  onManualSetup,
  error,
  setup,
  qrDataUrl,
  botUsername,
  detectedOwnerId,
  allowedIds,
  setAllowedIds,
  newAllowedId,
  setNewAllowedId,
  addAllowedId,
  apply,
  cancel,
  expiresIn
}: TelegramSetupViewProps) {
  return (
    <div className="rounded-sm border border-border bg-background/35 p-4">
      <div className="grid gap-1">
        <span className="font-mondwest text-sm text-foreground">Choose how to connect your Telegram bot</span>
        <span className="text-xs text-muted-foreground">
          Both options connect a bot you control and save its credentials only to this Hermes installation.
        </span>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2 sm:divide-x sm:divide-border">
        <div className="grid content-start gap-3 sm:pr-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium uppercase text-foreground">Quick setup</span>
            <Badge tone="success">recommended</Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            Scan a QR code and confirm in Telegram. Hermes creates the bot and detects your Telegram user ID
            automatically.
          </p>
          <Button
            size="sm"
            className="w-fit uppercase"
            onClick={() => void start()}
            disabled={phase !== 'idle'}
            prefix={phase === 'starting' ? <Spinner /> : <QrCode className="h-4 w-4" />}
          >
            {phase === 'starting' ? 'Starting…' : 'Create with QR'}
          </Button>
        </div>

        <div className="grid content-start gap-3 border-t border-border pt-4 sm:border-t-0 sm:pl-4 sm:pt-0">
          <span className="text-xs font-medium uppercase text-foreground">Use your own bot</span>
          <p className="text-xs text-muted-foreground">
            Create a bot with @BotFather, or connect one you already have, by entering its token and choosing who can
            use it.
          </p>
          <Button
            size="sm"
            outlined
            className="w-fit uppercase"
            onClick={onManualSetup}
            disabled={phase !== 'idle'}
            prefix={<Bot className="h-4 w-4" />}
          >
            Manual setup
          </Button>
        </div>
      </div>

      {platform.configured && (
        <div className="mt-4 border-t border-border pt-3 text-xs text-muted-foreground">
          Telegram credentials are already configured. A new QR setup or bot token will replace the current bot when you
          save.
        </div>
      )}

      {phase !== 'idle' && (
        <div className="mt-4 border-t border-border pt-4">
          <span className="text-xs text-muted-foreground">
            Finish or cancel the current QR setup before switching methods.
          </span>
        </div>
      )}

      {error && (
        <div className="mt-3 border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {setup && (
        <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px]">
          <div className="grid gap-3">
            {(phase === 'ready' || phase === 'applying') && (
              <div className="grid gap-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="success">Ready</Badge>
                  {botUsername && <span className="font-courier text-sm text-muted-foreground">@{botUsername}</span>}
                </div>

                <div className="grid gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Allowed users</span>
                    {detectedOwnerId && allowedIds.includes(detectedOwnerId) && (
                      <Badge tone="success">owner detected</Badge>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {allowedIds.map(id => (
                      <button
                        key={id}
                        type="button"
                        className="inline-flex items-center gap-1 border border-border px-2 py-1 font-courier text-xs text-foreground hover:border-destructive/50"
                        onClick={() => setAllowedIds(ids => ids.filter(existing => existing !== id))}
                      >
                        {id}
                        <X className="h-3 w-3" />
                      </button>
                    ))}
                    {allowedIds.length === 0 && (
                      <span className="text-sm text-muted-foreground">Add at least one Telegram user ID.</span>
                    )}
                  </div>
                </div>

                <div className="flex flex-col gap-2 sm:flex-row">
                  <Input
                    value={newAllowedId}
                    onChange={event => setNewAllowedId(event.target.value)}
                    placeholder="Telegram user ID"
                    className="font-courier"
                  />
                  <Button size="sm" outlined onClick={addAllowedId} prefix={<Check />}>
                    Add
                  </Button>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    className="uppercase"
                    onClick={() => void apply()}
                    disabled={phase === 'applying'}
                    prefix={phase === 'applying' ? <Spinner /> : <Save className="h-4 w-4" />}
                  >
                    {phase === 'applying' ? 'Saving…' : 'Save and restart'}
                  </Button>
                  <Button size="sm" ghost onClick={() => void cancel()}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </div>

          <div className="flex flex-col items-center justify-center gap-3">
            {qrDataUrl && <img src={qrDataUrl} alt="Telegram setup QR code" className="h-56 w-56 bg-white p-2" />}
            <p className="text-xs text-muted-foreground">
              Setup {setup.pairing_id.slice(-6)}: open Telegram, create your bot, then choose “Connect this bot”.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
              <Badge tone={expiresIn === 'expired' ? 'destructive' : 'outline'}>{expiresIn}</Badge>
              {phase === 'waiting' && <Badge tone="warning">waiting</Badge>}
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              <a
                href={setup.deep_link}
                target="_blank"
                rel="noreferrer"
                className="inline-flex h-8 items-center gap-1 border border-border px-3 text-xs uppercase text-foreground hover:border-foreground/40"
              >
                <ExternalLink className="h-4 w-4" />
                Open Telegram
              </a>
              <Button size="sm" ghost onClick={() => void cancel()}>
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
