import { useEffect, useRef, useState } from 'react'
import { createWisdomMuteController, initialWisdomMuteState, type WisdomMuteDuration } from '@hermes/shared'
import { useI18n } from '@/i18n'
import { api } from '@/lib/api'
import { Button } from '@nous-research/ui/ui/components/button'
import { Volume2, VolumeX, RefreshCw, Loader2 } from 'lucide-react'

export function WisdomNotificationSettings({ profile }: { profile?: string }) {
  return <Settings key={profile || 'default'} profile={profile} />
}

function Settings({ profile }: { profile?: string }) {
  const { t, locale } = useI18n()
  const copy = t.skills.wisdom.notificationPreferences
  const [open, setOpen] = useState(false)
  const [state, setState] = useState(initialWisdomMuteState)
  const [scope] = useState(profile)
  const controller = useRef<ReturnType<typeof createWisdomMuteController> | null>(null)

  useEffect(() => {
    if (!open) return
    const next = createWisdomMuteController({
      prepare: () => api.prepareWisdomMute(scope),
      read: () => api.getWisdomMute(scope),
      choose: (id, duration) => api.chooseWisdomMute(id, duration, scope),
      changed: setState
    })
    controller.current = next
    void next.refresh()
    const poll = () => {
      if (document.visibilityState !== 'hidden') void next.poll()
    }
    const timer = window.setInterval(poll, 15_000)
    window.addEventListener('focus', poll)
    return () => {
      next.dispose()
      controller.current = null
      window.clearInterval(timer)
      window.removeEventListener('focus', poll)
    }
  }, [open, scope])

  const muted = state.snapshot?.mute?.muted
  const until = state.snapshot?.mute?.muted_until
  const sync = state.snapshot?.sync?.preference_sync
  const notice =
    sync === 'pending' || sync === 'syncing'
      ? copy.pending
      : sync === 'failed'
        ? copy.failed
        : sync === 'conflict'
          ? copy.conflict
          : sync === 'expired'
            ? copy.expired
            : null
  const Icon = muted ? VolumeX : Volume2
  const choice = state.choice === undefined ? '' : (state.choice ?? 'off')

  return (
    <section className="border-b border-border min-w-0 py-2">
      <button
        type="button"
        className="flex max-w-full items-center gap-2 text-sm"
        aria-expanded={open}
        onClick={() => setOpen(value => !value)}
      >
        <Icon className="size-4 shrink-0" aria-hidden="true" />
        <span className="break-words text-start">{copy.title}</span>
      </button>
      {open && (
        <div className="mt-3 min-w-0 space-y-3">
          <p className="text-text-secondary text-xs">{copy.scope}</p>
          <p className="break-words text-sm" role="status">
            {state.snapshot?.mute
              ? muted
                ? copy.muted
                : copy.on
              : state.busy
                ? t.common.loading
                : t.skills.wisdom.unavailable}
            {muted && until && (
              <>
                {' '}
                · <time dateTime={until}>{new Date(until).toLocaleString(locale)}</time>
              </>
            )}
            {muted && state.snapshot?.mute?.forever && <> · {copy.forever}</>}
          </p>
          {notice && (
            <p className="text-xs" role="status">
              {notice}
            </p>
          )}
          {state.error && (
            <p className="text-red-500 text-xs" role="alert">
              {t.skills.wisdom.unavailable}
            </p>
          )}
          <form
            className="flex min-w-0 flex-wrap items-center gap-2"
            onSubmit={event => {
              event.preventDefault()
              void controller.current?.apply()
            }}
          >
            <select
              aria-label={copy.title}
              value={choice}
              disabled={state.busy || !state.control || state.retryingChoice}
              className="border-border bg-bg-primary h-9 min-w-0 max-w-full flex-1 rounded border px-2 text-sm"
              onChange={event =>
                controller.current?.select(
                  event.target.value === 'off' ? null : (event.target.value as WisdomMuteDuration)
                )
              }
            >
              <option value="" disabled>
                {copy.title}
              </option>
              <option value="off">{copy.on}</option>
              <option value="1_day">{copy.day}</option>
              <option value="1_week">{copy.week}</option>
              <option value="30_days">{copy.month}</option>
              <option value="forever">{copy.forever}</option>
            </select>
            <Button
              className="h-9 shrink-0"
              type="button"
              size="sm"
              outlined
              disabled={state.busy}
              aria-label={t.common.refresh}
              title={t.common.refresh}
              onClick={() => void controller.current?.refresh()}
            >
              <RefreshCw className="size-4" aria-hidden="true" />
            </Button>
            <Button
              className="h-9 shrink-0"
              type="submit"
              size="sm"
              disabled={state.busy || (!state.retryingChoice && (state.choice === undefined || !state.control))}
            >
              {state.busy && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
              {state.retryingChoice ? t.common.retry : t.common.save}
            </Button>
          </form>
        </div>
      )}
    </section>
  )
}
