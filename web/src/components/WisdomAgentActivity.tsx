import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { useI18n } from '@/i18n'
import { WisdomReviewTables } from './WisdomChecks'

export function WisdomAgentActivity({ profile }: { profile?: string }) {
  const { t } = useI18n()
  const [activity, setActivity] = useState<Awaited<ReturnType<typeof api.getWisdomMediation>> | null>(null)
  useEffect(() => {
    let active = true
    let revision = 0
    setActivity(null)
    const refresh = async () => {
      const request = ++revision
      try {
        const result = await api.getWisdomMediation(profile)
        if (active && request === revision) setActivity(result)
      } catch {
        // Passive reads never clear previously delivered advice on an outage.
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 30_000)
    return () => { active = false; window.clearInterval(timer) }
  }, [profile])
  if (!activity) return null
  const entries = activity.assessments.filter(entry => entry.advice)
  if (!entries.length) return null
  return <section aria-label={t.skills.wisdom.title} className="border-y border-border py-3">
    <h3 className="text-sm font-semibold">{t.skills.wisdom.title}</h3>
    {entries.map(entry => <article key={entry.id} className="min-w-0 border-t border-border py-3">
      <h4 className="break-words text-sm font-medium">{entry.advice?.title}</h4>
      <p className="mt-1 whitespace-pre-wrap break-words text-sm text-text-secondary">{entry.advice?.explanation}</p>
      {activity.interactions.filter(item => item.assessment_id === entry.id).map(item =>
        <div key={item.id} className="mt-2 min-w-0">
          <p className="text-xs font-medium">{item.facts.editorial_name || item.facts.slug}{item.facts.version ? ` · v${item.facts.version}` : ''}</p>
          <p className="mt-1 text-xs text-text-secondary">{item.operation}: {item.state.replaceAll('_', ' ')}</p>
          {item.facts.compatibility && <p className="text-xs text-text-secondary">{item.facts.compatibility.outcome.replaceAll('_', ' ')}</p>}
          {item.facts.modified && <p className="text-xs text-red-400">{t.skills.wisdom.unsavedChanges}</p>}
          {item.facts.sensitive_expansion?.map((warning, index) => <p key={index} className="break-words text-xs text-red-400">{warning}</p>)}
          <details className="mt-2">
            <summary className="cursor-pointer text-xs">{t.skills.wisdom.reviewExact}</summary>
            <WisdomReviewTables security={item.facts.security_check} professionalism={item.facts.professionalism_check} />
          </details>
        </div>
      )}
    </article>)}
  </section>
}
