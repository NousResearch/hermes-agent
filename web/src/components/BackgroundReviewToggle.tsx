import { useState } from 'react'
import { Switch } from '@nous-research/ui/ui/components/switch'
import { api } from '@/lib/api'
import { errorMessage } from '@/lib/api-error'

interface BackgroundReviewToggleProps {
  enabled: boolean
  onSaved(): void
}

export function BackgroundReviewToggle({ enabled, onSaved }: BackgroundReviewToggleProps) {
  const [state, setState] = useState({ enabled, value: enabled })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  if (state.enabled !== enabled) {
    setState({ enabled, value: enabled })
  }

  const save = async (next: boolean) => {
    const previous = state.value
    setState(current => ({ ...current, value: next }))
    setBusy(true)
    setError('')
    try {
      await api.saveConfig({ auxiliary: { background_review: { enabled: next } } })
      onSaved()
    } catch (err) {
      setState(current => (current.enabled === enabled ? { ...current, value: previous } : current))
      setError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="max-w-64 text-xs text-text-secondary">
      <label className="flex items-center gap-2">
        <Switch aria-label="Background review" checked={state.value} disabled={busy} onCheckedChange={save} />
        Automatic review
      </label>
      <p>Opt in to memory and skill learning after turns. Uses model tokens.</p>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}
    </div>
  )
}
