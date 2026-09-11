import { createWisdomSyncController, initialWisdomSyncView, wisdomSyncCopy, type WisdomSyncState } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'
import { Button } from '@nous-research/ui/ui/components/button'
import { Loader2, RefreshCw } from 'lucide-react'
import { api } from '@/lib/api'
import { useI18n } from '@/i18n'

export function WisdomSyncStatus({ profile }: { profile?: string }) {
  return <SyncStatus key={profile || 'default'} profile={profile} />
}

function SyncStatus({ profile }: { profile?: string }) {
  const { t } = useI18n()
  const copy = t.skills.wisdom.syncRecovery ?? wisdomSyncCopy
  const [open, setOpen] = useState(false)
  const [scope] = useState(profile)
  const [view, setView] = useState(initialWisdomSyncView)
  const controller = useRef<ReturnType<typeof createWisdomSyncController> | null>(null)

  useEffect(() => {
    if (!open) return
    const next = createWisdomSyncController({
      read: () => api.getWisdomSync(scope),
      retry: () => api.retryWisdomSync(scope),
      changed: setView
    })
    controller.current = next
    void next.refresh()
    const poll = () => {
      if (document.visibilityState !== 'hidden') void next.refresh()
    }
    const timer = window.setInterval(poll, 15_000)
    return () => {
      next.dispose()
      controller.current = null
      window.clearInterval(timer)
    }
  }, [open, scope])
  return (
    <section className="min-w-0 border-b border-border py-2">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(value => !value)}
        className="flex max-w-full items-center gap-2 text-sm"
      >
        <RefreshCw aria-hidden="true" className="size-4 shrink-0" />
        <span className="break-words text-start">{copy.title}</span>
      </button>
      {open && (
        <div className="mt-3 min-w-0 space-y-3">
          <p className="text-xs text-text-secondary">{copy.scope}</p>
          {view.error && (
            <p role="alert" className="text-xs text-red-500">
              {copy.unavailable}
            </p>
          )}
          {!view.snapshot && view.busy && (
            <p role="status" className="text-xs">
              {t.common.loading}
            </p>
          )}
          {view.snapshot && !view.error && (
            <dl className="space-y-2 text-xs" aria-live="polite">
              {(['delivery', 'operation'] as const).map(kind => {
                const counts = view.snapshot![kind]
                const states = (Object.keys(counts) as WisdomSyncState[]).filter(state => counts[state] > 0)
                return (
                  <div key={kind}>
                    <dt className="font-medium">{copy[kind]}</dt>
                    <dd className="break-words text-text-secondary">
                      {states.length
                        ? states.map(state => (
                            <div key={state}>
                              {copy.states[state]}: {counts[state]}
                            </div>
                          ))
                        : copy.current}
                    </dd>
                  </div>
                )
              })}
            </dl>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              outlined
              title={t.common.refresh}
              aria-label={t.common.refresh}
              disabled={view.busy}
              onClick={() => void controller.current?.refresh()}
            >
              <RefreshCw className="size-4" aria-hidden="true" />
            </Button>
            <Button
              type="button"
              size="sm"
              disabled={view.busy || view.error || !view.snapshot?.can_retry}
              onClick={() => void controller.current?.retry()}
            >
              {view.busy && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
              {copy.retry}
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
