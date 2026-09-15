import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { fuzzyRank, fuzzyScoreMulti } from '@hermes/shared/fuzzy'
import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared/gateway-events'
import { modelSearchText } from '@hermes/shared/model-search-text'
import { REASONING_EFFORTS } from '@hermes/shared/reasoning-effort'
import { useEffect, useMemo, useState } from 'react'

import { providerDisplayNames } from '../domain/providers.js'
import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, useOverlayKeys, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 12
const MIN_WIDTH = 40
const MAX_WIDTH = 90

type Stage = 'hop' | 'provider' | 'key' | 'model' | 'reasoning' | 'disconnect'

type ProviderRow = { name: string; provider: ModelOptionProvider }

export type ModelHopRow = {
  model: string
  name: string
  provider: ModelOptionProvider
  selector: string
}

/** Flat catalog rows for the omp `/switch` hop: `provider/id` searchable. */
export function buildModelHopRows(providers: ModelOptionProvider[], names: string[]): ModelHopRow[] {
  const rows: ModelHopRow[] = []

  providers.forEach((provider, i) => {
    const name = names[i] ?? provider.name ?? provider.slug

    for (const model of provider.models ?? []) {
      rows.push({
        model,
        name,
        provider,
        selector: `${provider.slug}/${model}`
      })
    }
  })

  return rows
}

export function hopRowSearchText(row: ModelHopRow): string {
  // Rank on `provider/id` (OMP /switch). Extra display names made `son4` hit `hermes-4`.
  return `${row.selector} ${modelSearchText(row.model)}`
}

export function hopIsCurrent(h: ModelHopRow, current: string) {
  return h.selector === current || (!!h.provider.is_current && h.model === current)
}

export function hopCurrentIndex(rows: ModelHopRow[], current: string) {
  const i = rows.findIndex(h => hopIsCurrent(h, current))
  return i < 0 ? 0 : i
}

/** Printable paste (or one typed char). Drops controls so Tab/Esc/newlines never enter the filter. */
export function searchAppend(prev: string, ch: string) {
  let add = ''
  for (const c of ch) {
    if (c >= ' ') add += c
  }
  return add ? prev + add : prev
}

/** Rank/paint query after a `nous/` provider scope. `nous/` → empty (no fake hits on the slug). */
export function hopPaintQuery(q: string) {
  const t = q.trim()
  const i = t.indexOf('/')
  return i < 0 ? t : t.slice(i + 1).trim()
}

export function keepReasoningLabel(current: string) {
  const v = current.trim().toLowerCase()
  if (!v || v === 'hide' || v === 'show') return 'Keep current effort'
  return `Keep current effort (${v})`
}

/** Clamp a list index. Empty list stays 0; page/home/end use a large |delta|. */
export function listStep(sel: number, n: number, delta: number) {
  if (n <= 0) return 0
  return Math.max(0, Math.min(n - 1, sel + delta))
}

export function hopLocked(h: ModelHopRow) {
  return (h.provider.unavailable_models ?? []).includes(h.model)
}

/** Compact in/out $/M from inventory pricing. Empty when unknown. */
export function hopPrice(h: ModelHopRow) {
  const p = h.provider.pricing?.[h.model]
  if (!p) return ''
  if (p.free) return 'free'
  const sale = typeof p.discount_percent === 'number' ? ` -${p.discount_percent}%` : ''
  return `${p.input || '?'}/${p.output || '?'}${sale}`
}

export function hopDetail(h?: ModelHopRow) {
  if (!h) return ''
  const bits: string[] = []
  const price = hopPrice(h)
  if (price) bits.push(price)
  if (h.provider.capabilities?.[h.model]?.fast) bits.push('fast')
  if (hopLocked(h)) bits.push('locked')
  return bits.join(' · ')
}

/** Current, then per-provider featured, then catalog order (OMP recents analogue). */
export function orderHopRows(rows: ModelHopRow[], current: string) {
  const at = new Map(rows.map((h, i) => [h.selector, i]))
  const feat = (h: ModelHopRow) => (h.provider.featured_models ?? []).includes(h.model)
  return [...rows].sort((a, b) => {
    const ra = hopIsCurrent(a, current) ? 0 : feat(a) ? 1 : 2
    const rb = hopIsCurrent(b, current) ? 0 : feat(b) ? 1 : 2
    return ra - rb || (at.get(a.selector) ?? 0) - (at.get(b.selector) ?? 0)
  })
}

/** Consecutive runs of `text` whose indices fuzzy-matched `q` (OMP /switch). */
export function paintHits(text: string, q: string): { t: string; hit: boolean }[] {
  const hits = new Set(fuzzyScoreMulti(text, q.trim())?.positions ?? [])
  const out: { t: string; hit: boolean }[] = []

  for (let i = 0; i < text.length; i++) {
    const hit = hits.has(i)
    const last = out.at(-1)

    if (last && last.hit === hit) {
      last.t += text[i]
    } else {
      out.push({ t: text[i]!, hit })
    }
  }

  return out
}

export function filterModelHopRows(rows: ModelHopRow[], query: string): ModelHopRow[] {
  const q = query.trim()
  if (!q) {
    return rows
  }

  // `nous/` is a provider scope (OMP /switch). The rest is `@hermes/shared/fuzzy`.
  const slash = q.indexOf('/')
  let pool = rows
  let rest = q
  if (slash >= 0) {
    const providerQuery = q.slice(0, slash).trim().toLowerCase()
    rest = q.slice(slash + 1)
    if (providerQuery) {
      pool = rows.filter(
        row =>
          row.provider.slug.toLowerCase().startsWith(providerQuery) ||
          (row.provider.name ?? '').toLowerCase().startsWith(providerQuery)
      )
    }
    if (!rest.trim()) {
      return pool
    }
  }

  return fuzzyRank(pool, rest, hopRowSearchText).map(r => r.item)
}


/** Rows of the effort step (step 3/3): the shared ladder, the off state, then
 *  "keep current" (empty value = no `--reasoning` flag on the emitted command). */
export const REASONING_PICKER_ROWS: ReadonlyArray<{ label: string; value: string }> = [
  ...REASONING_EFFORTS.map(level => ({ label: level, value: level })),
  { label: 'none (disable reasoning)', value: 'none' },
  { label: 'Keep current effort', value: '' }
]

export const KEEP_REASONING_IDX = REASONING_PICKER_ROWS.length - 1

/** False only when the catalog says the picked model has no reasoning control;
 *  unknown capabilities keep the step (a no-op dial beats hiding a real one). */
export function pickerOffersReasoning(provider: ModelOptionProvider | undefined, model: string): boolean {
  return provider?.capabilities?.[model]?.reasoning !== false
}

/** The `/model` argument the picker emits: model + provider + scope, plus
 *  `--reasoning <level>` when an effort was picked. */
export function modelPickerCommand(
  model: string,
  providerSlug: string,
  persistGlobal: boolean,
  reasoning = ''
): string {
  const scope = persistGlobal ? '--global' : TUI_SESSION_MODEL_FLAG
  const effort = reasoning ? ` --reasoning ${reasoning}` : ''

  return `${model} --provider ${providerSlug}${effort} ${scope}`
}

export function providerIndexAfterClearingFilter(
  providerRows: ProviderRow[],
  provider: ModelOptionProvider | undefined
) {
  if (!provider) {
    return -1
  }

  return providerRows.findIndex(row => row.provider.slug === provider.slug)
}

function HitLabel({ dim = 0, q, t, text }: { dim?: number; q: string; t: Theme; text: string }) {
  let n = 0
  return (
    <>
      {paintHits(text, q).map((p, i) => {
        const start = n
        n += p.t.length
        const color = p.hit ? t.color.accent : start < dim ? t.color.label : undefined
        return color ? (
          <Text key={i} color={color}>
            {p.t}
          </Text>
        ) : (
          p.t
        )
      })}
    </>
  )
}

export function ModelPicker({
  allowPersistGlobal = true,
  gw,
  initialRefresh = false,
  initialStage = 'hop',
  maxWidth,
  onCancel,
  onSelect,
  sessionId,
  t
}: ModelPickerProps) {
  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [currentModel, setCurrentModel] = useState('')
  const [currentReasoning, setCurrentReasoning] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const [persistGlobal, setPersistGlobal] = useState(false)
  const [providerIdx, setProviderIdx] = useState(0)
  const [modelIdx, setModelIdx] = useState(0)
  const [reasoningIdx, setReasoningIdx] = useState(KEEP_REASONING_IDX)
  // Model chosen on step 2, awaiting the effort pick on step 3.
  const [pendingModel, setPendingModel] = useState('')
  // Hop Enter that offers reasoning must Esc back to the hop catalog, not the
  // wizard's provider-scoped model list (same emit path as the model stage).
  const [reasoningOrigin, setReasoningOrigin] = useState<'hop' | 'model'>('model')
  const [stage, setStage] = useState<Stage>(initialStage === 'provider' ? 'provider' : 'hop')
  const [keyInput, setKeyInput] = useState('')
  const [keySaving, setKeySaving] = useState(false)
  const [keyError, setKeyError] = useState('')
  // Type-to-filter query, scoped per stage (cleared on stage change).
  const [filter, setFilter] = useState('')

  const { stdout } = useStdout()
  // Pin the picker to a stable width so the FloatBox parent (which shrinks-
  // to-fit with alignSelf="flex-start") doesn't resize as long provider /
  // model names scroll into view, and so `wrap="truncate-end"` on each row
  // has an actual constraint to truncate against. Optional maxWidth lets
  // grid layouts hand the picker its cell budget.
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

  useEffect(() => {
    gw.request<ModelOptionsResult>('model.options', {
      ...(sessionId ? { session_id: sessionId } : {}),
      ...(initialRefresh ? { refresh: true } : {}),
      // The TUI picker shows the full provider universe with setup
      // affordances ("paste KEY to activate"), so opt into unconfigured
      // rows — the backend now defaults to the configured subset for
      // desktop chat pickers (#56974).
      include_unconfigured: true
    })
      .then(raw => {
        const r = asRpcResult<ModelOptionsResult>(raw)

        if (!r) {
          setErr('invalid response: model.options')
          setLoading(false)

          return
        }

        const next = r.providers ?? []
        setProviders(next)
        setCurrentModel(String(r.model ?? ''))
        gw.request<{ value?: string }>('config.get', {
          key: 'reasoning',
          ...(sessionId ? { session_id: sessionId } : {})
        })
          .then(raw => {
            const effort = asRpcResult<{ value?: string }>(raw)
            setCurrentReasoning(String(effort?.value ?? ''))
          })
          .catch(() => setCurrentReasoning(''))
        setProviderIdx(
          Math.max(
            0,
            next.findIndex(p => p.is_current)
          )
        )
        setModelIdx(hopCurrentIndex(buildModelHopRows(next, providerDisplayNames(next)), String(r.model ?? '')))
        setStage(initialStage === 'provider' ? 'provider' : 'hop')
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }, [gw, initialRefresh, sessionId])

  const names = useMemo(() => providerDisplayNames(providers), [providers])

  // Provider rows carry their display name so fuzzy filtering can match on
  // name + slug while keeping the name/provider pairing intact across ranking.
  const providerRows = useMemo(
    () => providers.map((p, i) => ({ provider: p, name: names[i] ?? p.name ?? p.slug })),
    [providers, names]
  )

  // providerIdx / modelIdx always index into the *displayed* (filtered) lists.
  // With an empty filter the filtered list equals the full list, so navigation
  // behaves exactly as before. Filtering only applies on the relevant stage.
  const filteredProviderRows = useMemo(() => {
    if (stage !== 'provider' || !filter.trim()) {
      return providerRows
    }

    return fuzzyRank(
      providerRows,
      filter,
      row => `${row.name} ${row.provider.slug} ${(row.provider.models ?? []).join(' ')}`
    ).map(r => r.item)
  }, [providerRows, filter, stage])

  const hopRows = useMemo(() => buildModelHopRows(providers, names), [providers, names])

  const filteredHopRows = useMemo(() => {
    if (stage !== 'hop') {
      return hopRows
    }
    if (!filter.trim()) {
      return orderHopRows(hopRows, currentModel)
    }

    return filterModelHopRows(hopRows, filter)
  }, [hopRows, filter, stage, currentModel])

  const provider = filteredProviderRows[providerIdx]?.provider
  const allModels = useMemo(() => provider?.models ?? [], [provider])

  const filteredModels = useMemo(() => {
    if (stage !== 'model' || !filter.trim()) {
      return allModels
    }

    // modelSearchText adds aliases for brand-less wire ids (e.g. Kimi
    // Coding `k3` still matches a "kimi" query).
    return fuzzyRank(allModels, filter, modelSearchText).map(r => r.item)
  }, [allModels, filter, stage])

  const models = filteredModels

  // Keep the active selection within the (possibly filtered) list bounds.
  useEffect(() => {
    if (providerIdx >= filteredProviderRows.length && filteredProviderRows.length > 0) {
      setProviderIdx(0)
    }
  }, [filteredProviderRows.length, providerIdx])

  useEffect(() => {
    if (modelIdx >= models.length && models.length > 0) {
      setModelIdx(0)
    }
  }, [models.length, modelIdx])

  useEffect(() => {
    if (stage === 'hop' && modelIdx >= filteredHopRows.length && filteredHopRows.length > 0) {
      setModelIdx(0)
    }
  }, [filteredHopRows.length, modelIdx, stage])

  useEffect(() => {
    if (stage !== 'hop' || loading || hopRows.length > 0) {
      return
    }

    setStage('provider')
  }, [hopRows.length, loading, stage])

  const back = () => {
    // Esc first clears an active filter on the list stages, before navigating.
    if ((stage === 'provider' || stage === 'model' || stage === 'hop') && filter.trim()) {
      // Preserve the selected provider across filter clear (same fix as
      // Enter→key/model and Ctrl+D transitions above).
      const fullProviderIdx = providerIndexAfterClearingFilter(providerRows, provider)

      if (fullProviderIdx >= 0) {
        setProviderIdx(fullProviderIdx)
      } else if (stage === 'provider') {
        setProviderIdx(0)
      }

      setFilter('')
      setModelIdx(0)

      return
    }

    if (stage === 'reasoning') {
      setStage(reasoningOrigin)
      setPendingModel('')
      setReasoningIdx(KEEP_REASONING_IDX)

      return
    }

    if (stage === 'model' || stage === 'key' || stage === 'disconnect') {
      setStage('provider')
      setModelIdx(0)
      setKeyInput('')
      setKeyError('')
      setKeySaving(false)
      setFilter('')

      return
    }

    onCancel()
  }

  // On the list stages we capture printable keys (including 'q') into the
  // filter, so the shared overlay q/Esc handler must yield to our own handler.
  const listStage = stage === 'hop' || stage === 'provider' || stage === 'model' || stage === 'reasoning'
  useOverlayKeys({ disabled: listStage, onBack: back, onClose: onCancel })

  useInput((ch, key) => {
    // Loading/error/empty: Esc/q close. Never absorb type/Enter into a hidden hop filter.
    if (loading || err || !providers.length) {
      if (key.escape || ch === 'q') onCancel()
      return
    }

    // Key entry stage handles its own input
    if (stage === 'key') {
      if (keySaving) {
        return
      }

      if (key.return) {
        if (!keyInput.trim()) {
          return
        }

        setKeySaving(true)
        setKeyError('')
        gw.request<{ provider?: ModelOptionProvider }>('model.save_key', {
          slug: provider?.slug,
          api_key: keyInput.trim(),
          ...(sessionId ? { session_id: sessionId } : {})
        })
          .then(raw => {
            const r = asRpcResult<{ provider?: ModelOptionProvider }>(raw)

            if (!r?.provider) {
              setKeyError('failed to save key')
              setKeySaving(false)

              return
            }

            // Update the provider in our list with fresh data
            setProviders(prev => prev.map(p => (p.slug === r.provider!.slug ? r.provider! : p)))
            setKeyInput('')
            setKeySaving(false)
            setStage('model')
            setModelIdx(0)
          })
          .catch((e: unknown) => {
            setKeyError(rpcErrorMessage(e))
            setKeySaving(false)
          })

        return
      }

      if (key.backspace || key.delete) {
        setKeyInput(v => v.slice(0, -1))

        return
      }

      // ctrl+u clears input
      if (ch === '\u0015') {
        setKeyInput('')

        return
      }

      if (ch && !key.ctrl && !key.meta) {
        setKeyInput(v => v + ch)
      }

      return
    }

    // Disconnect confirmation stage
    if (stage === 'disconnect') {
      if (ch.toLowerCase() === 'y' || key.return) {
        if (!provider) {
          setStage('provider')

          return
        }

        setKeySaving(true)
        gw.request<{ disconnected?: boolean }>('model.disconnect', {
          slug: provider.slug,
          ...(sessionId ? { session_id: sessionId } : {})
        })
          .then(raw => {
            const r = asRpcResult<{ disconnected?: boolean }>(raw)

            if (r?.disconnected) {
              // Mark provider as unauthenticated in local state
              setProviders(prev =>
                prev.map(p =>
                  p.slug === provider.slug
                    ? {
                        ...p,
                        authenticated: false,
                        models: [],
                        total_models: 0,
                        warning: p.key_env ? `paste ${p.key_env} to activate` : 'run `hermes model` to configure'
                      }
                    : p
                )
              )
            }

            setKeySaving(false)
            setStage('provider')
          })
          .catch(() => {
            setKeySaving(false)
            setStage('provider')
          })

        return
      }

      if (ch.toLowerCase() === 'n' || key.escape) {
        setStage('provider')

        return
      }

      return
    }

    // Effort stage (step 3/3): plain arrow list, no filter.
    if (stage === 'reasoning') {
      if (key.escape) {
        back()

        return
      }

      if (ch === 'q') {
        onCancel()

        return
      }

      if (key.upArrow && reasoningIdx > 0) {
        setReasoningIdx(v => v - 1)

        return
      }

      if (key.downArrow && reasoningIdx < REASONING_PICKER_ROWS.length - 1) {
        setReasoningIdx(v => v + 1)

        return
      }

      const rn = REASONING_PICKER_ROWS.length
      if (key.pageUp || key.wheelUp) {
        setReasoningIdx(v => listStep(v, rn, key.pageUp ? -VISIBLE : -1))
        return
      }
      if (key.pageDown || key.wheelDown) {
        setReasoningIdx(v => listStep(v, rn, key.pageDown ? VISIBLE : 1))
        return
      }
      if (key.home) {
        setReasoningIdx(0)
        return
      }
      if (key.end) {
        setReasoningIdx(KEEP_REASONING_IDX)
        return
      }

      if (allowPersistGlobal && key.ctrl && ch === 'g') {
        setPersistGlobal(v => !v)

        return
      }

      if (key.return && provider && pendingModel) {
        onSelect(
          modelPickerCommand(
            pendingModel,
            provider.slug,
            allowPersistGlobal && persistGlobal,
            REASONING_PICKER_ROWS[reasoningIdx]?.value ?? ''
          )
        )
      }

      return
    }

    // List-stage Esc/q handling (overlay keys are disabled while on a list
    // stage so 'q' can be typed into the filter).
    if (key.escape) {
      back()

      return
    }

    if (ch === 'q' && !filter) {
      onCancel()

      return
    }

    const count =
      stage === 'hop' ? filteredHopRows.length : stage === 'provider' ? filteredProviderRows.length : models.length
    const sel = stage === 'provider' ? providerIdx : modelIdx
    const setSel = stage === 'provider' ? setProviderIdx : setModelIdx

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)

      return
    }

    if (key.downArrow && sel < count - 1) {
      setSel(v => v + 1)

      return
    }

    if (key.pageUp || key.wheelUp) {
      setSel(v => listStep(v, count, key.pageUp ? -VISIBLE : -1))
      return
    }
    if (key.pageDown || key.wheelDown) {
      setSel(v => listStep(v, count, key.pageDown ? VISIBLE : 1))
      return
    }
    if (key.home) {
      setSel(0)
      return
    }
    if (key.end) {
      setSel(v => listStep(v, count, count))
      return
    }

    if (key.return) {
      if (stage === 'hop') {
        const hop = filteredHopRows[modelIdx]

        if (!hop || hopLocked(hop)) {
          return
        }

        if (pickerOffersReasoning(hop.provider, hop.model)) {
          const fullProviderIdx = providerIndexAfterClearingFilter(providerRows, hop.provider)

          if (fullProviderIdx >= 0) {
            setProviderIdx(fullProviderIdx)
          }

          setPendingModel(hop.model)
          setReasoningIdx(KEEP_REASONING_IDX)
          setReasoningOrigin('hop')
          setStage('reasoning')
        } else {
          onSelect(modelPickerCommand(hop.model, hop.provider.slug, allowPersistGlobal && persistGlobal))
        }

        return
      }

      if (stage === 'provider') {
        if (!provider) {
          return
        }

        if (provider.authenticated === false) {
          // api_key providers: prompt for key inline
          if (provider.auth_type === 'api_key' && provider.key_env) {
            const fullProviderIdx = providerIndexAfterClearingFilter(providerRows, provider)

            if (fullProviderIdx >= 0) {
              setProviderIdx(fullProviderIdx)
            }

            setStage('key')
            setKeyInput('')
            setKeyError('')
            setFilter('')
          }

          // Other auth types: no-op (warning shown tells them to run hermes model)
          return
        }

        const fullProviderIdx = providerIndexAfterClearingFilter(providerRows, provider)

        if (fullProviderIdx >= 0) {
          setProviderIdx(fullProviderIdx)
        }

        setStage('model')
        setModelIdx(0)
        setFilter('')

        return
      }

      const model = models[modelIdx]

      if (provider && model) {
        if ((provider.unavailable_models ?? []).includes(model)) {
          return
        }
        if (pickerOffersReasoning(provider, model)) {
          // Step 3/3: effort for the picked model (skipped on reasoning-free routes).
          setPendingModel(model)
          setReasoningIdx(KEEP_REASONING_IDX)
          setReasoningOrigin('model')
          setStage('reasoning')
        } else {
          onSelect(modelPickerCommand(model, provider.slug, allowPersistGlobal && persistGlobal))
        }
      } else {
        setStage('provider')
      }

      return
    }

    // Backspace removes the last filter character; Esc (above) clears a
    // non-empty filter before navigating back.
    if (key.backspace || key.delete) {
      setFilter(v => v.slice(0, -1))
      setSel(0)

      return
    }

    // Ctrl+U clears the filter. (Ctrl held → ch is the key name 'u'.)
    if (key.ctrl && ch === 'u') {
      setFilter('')
      setSel(0)

      return
    }

    // Persist-global toggle moved to Ctrl+G so 'g' can be typed into the
    // filter. With Ctrl held, @hermes/ink reports `ch` as the key name ('g'),
    // not the raw control byte (see input-event.ts: input = ctrl ? name : seq).
    if (allowPersistGlobal && key.ctrl && ch === 'g') {
      setPersistGlobal(v => !v)

      return
    }

    // Disconnect (Ctrl+D): only in provider stage, only for authenticated providers.
    if (key.ctrl && ch === 'd' && stage === 'provider' && provider?.authenticated !== false) {
      const fullProviderIdx = providerIndexAfterClearingFilter(providerRows, provider)

      if (fullProviderIdx >= 0) {
        setProviderIdx(fullProviderIdx)
      }

      setStage('disconnect')
      setFilter('')

      return
    }

    const nav =
      key.tab ||
      key.return ||
      key.escape ||
      key.home ||
      key.end ||
      key.pageUp ||
      key.pageDown ||
      key.upArrow ||
      key.downArrow
    if (!key.ctrl && !key.meta && !nav && searchAppend('', ch)) {
      setFilter(v => searchAppend(v, ch))
      setSel(0)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading models…</Text>
  }

  if (err) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (!providers.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>no providers available</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  // ── Key entry stage ──────────────────────────────────────────────────
  if (stage === 'key' && provider) {
    const masked = keyInput ? '•'.repeat(Math.min(keyInput.length, 40)) : ''

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Configure {provider.name}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Paste your API key below (saved to ~/.hermes/.env)
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {provider.key_env}:
        </Text>

        <Text color={t.color.accent} wrap="truncate-end">
          {'  '}
          {masked || '(empty)'}
          {keySaving ? '' : '▎'}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        {keyError ? (
          <Text color={t.color.label} wrap="truncate-end">
            error: {keyError}
          </Text>
        ) : keySaving ? (
          <Text color={t.color.muted} wrap="truncate-end">
            saving…
          </Text>
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {' '}
          </Text>
        )}

        <OverlayHint t={t}>Enter save · Ctrl+U clear · Esc back</OverlayHint>
      </Box>
    )
  }

  // ── Disconnect confirmation stage ─────────────────────────────────────
  if (stage === 'disconnect' && provider) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Disconnect {provider.name}?
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          This removes saved credentials for {provider.name}.
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          You can re-authenticate later by selecting it again.
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        {keySaving ? (
          <Text color={t.color.muted} wrap="truncate-end">
            disconnecting…
          </Text>
        ) : (
          <OverlayHint t={t}>y/Enter confirm · n/Esc cancel</OverlayHint>
        )}
      </Box>
    )
  }

  // ── Session hop (omp /switch): flat provider/id list ──────────────────
  if (stage === 'hop') {
    const labels = filteredHopRows.map(row => row.selector)
    const { items, offset } = windowItems(labels, modelIdx, VISIBLE)
    const noMatches = !!filter.trim() && labels.length === 0

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Switch model
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Session-only · type provider/model · /model --provider for providers
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Current: {currentModel || '(unknown)'}
        </Text>
        <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
          {filter ? `filter: ${filter}▎` : 'type to search · ↑/↓ select'}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          {offset > 0 ? ` ↑ ${offset} more` : ' '}
        </Text>

        {noMatches ? (
          <Text color={t.color.muted} wrap="truncate-end">
            no models match
          </Text>
        ) : (
          Array.from({ length: VISIBLE }, (_, i) => {
            const row = items[i]
            const idx = offset + i
            const hop = filteredHopRows[idx]
            const current = hop ? hopIsCurrent(hop, currentModel) : false
            const locked = hop ? hopLocked(hop) : false
            const dim = hop ? hop.provider.slug.length + 1 : 0

            return row ? (
              <Text
                color={locked ? t.color.label : t.color.muted}
                {...chipRowProps(t, modelIdx === idx)}
                key={hop?.selector ?? `hop-${idx}`}
                wrap="truncate-end"
              >
                {modelIdx === idx ? '▸ ' : current ? '* ' : '  '}
                {idx + 1}. <HitLabel dim={dim} q={hopPaintQuery(filter)} t={t} text={row} />
              </Text>
            ) : (
              <Text color={t.color.muted} key={`pad-${i}`} wrap="truncate-end">
                {' '}
              </Text>
            )
          })
        )}

        <Text color={t.color.muted} wrap="truncate-end">
          {offset + VISIBLE < labels.length ? ` ↓ ${labels.length - offset - VISIBLE} more` : ' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {hopDetail(filteredHopRows[modelIdx]) || ' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          persist: {allowPersistGlobal ? (persistGlobal ? 'global' : 'session') : 'session'}
          {allowPersistGlobal ? ' · ^g toggle' : ' only'}
        </Text>
        <OverlayHint t={t}>↑/↓ select · Enter use · type nous/claude · Esc close</OverlayHint>
      </Box>
    )
  }

  // ── Provider selection stage ─────────────────────────────────────────
  if (stage === 'provider') {
    const rows = filteredProviderRows.map(({ provider: p, name }) => {
      const authMark = p.authenticated === false ? '○' : p.is_current ? '*' : '●'
      const modelCount = p.total_models ?? p.models?.length ?? 0

      const suffix =
        p.authenticated === false ? (p.auth_type === 'api_key' ? '(no key)' : '(needs setup)') : `${modelCount} models`

      return `${authMark} ${name} · ${suffix}`
    })

    const { items, offset } = windowItems(rows, providerIdx, VISIBLE)
    const noMatches = !!filter.trim() && rows.length === 0

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Select provider (step 1/3)
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Full model IDs on the next step · Enter to continue
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Current: {currentModel || '(unknown)'}
        </Text>
        <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
          {filter ? `filter: ${filter}▎` : 'type to filter · ↑/↓ select'}
        </Text>
        <Text color={t.color.label} wrap="truncate-end">
          {provider?.warning ? `warning: ${provider.warning}` : ' '}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          {offset > 0 ? ` ↑ ${offset} more` : ' '}
        </Text>

        {noMatches ? (
          <Text color={t.color.muted} wrap="truncate-end">
            no providers match
          </Text>
        ) : (
          Array.from({ length: VISIBLE }, (_, i) => {
            const row = items[i]
            const idx = offset + i
            const p = filteredProviderRows[idx]?.provider
            const dimmed = p?.authenticated === false

            return row ? (
              <Text
                color={dimmed ? t.color.label : t.color.muted}
                {...chipRowProps(t, providerIdx === idx)}
                key={p?.slug ?? `row-${idx}`}
                wrap="truncate-end"
              >
                {providerIdx === idx ? '▸ ' : '  '}
                {idx + 1}. <HitLabel q={filter} t={t} text={row} />
              </Text>
            ) : (
              <Text color={t.color.muted} key={`pad-${i}`} wrap="truncate-end">
                {' '}
              </Text>
            )
          })
        )}

        <Text color={t.color.muted} wrap="truncate-end">
          {offset + VISIBLE < rows.length ? ` ↓ ${rows.length - offset - VISIBLE} more` : ' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          persist: {allowPersistGlobal ? (persistGlobal ? 'global' : 'session') : 'session'}
          {allowPersistGlobal ? ' · ^g toggle' : ' only'}
        </Text>
        <OverlayHint t={t}>↑/↓ select · Enter choose · ^d disconnect · Esc clear/back · q close</OverlayHint>
      </Box>
    )
  }

  // ── Reasoning effort stage ───────────────────────────────────────────
  if (stage === 'reasoning') {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Reasoning effort (step 3/3)
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {pendingModel} · applies with the switch (same scope) · Esc back
        </Text>

        {REASONING_PICKER_ROWS.map((row, idx) => (
          <Text
            color={t.color.muted}
            {...chipRowProps(t, reasoningIdx === idx)}
            key={row.value || 'keep'}
            wrap="truncate-end"
          >
            {reasoningIdx === idx ? '▸ ' : '  '}
            {idx + 1}. {row.value === '' ? keepReasoningLabel(currentReasoning) : row.label}
          </Text>
        ))}

        <Text color={t.color.muted} wrap="truncate-end">
          persist: {allowPersistGlobal ? (persistGlobal ? 'global' : 'session') : 'session'}
          {allowPersistGlobal ? ' · ^g toggle' : ' only'}
        </Text>
        <OverlayHint t={t}>↑/↓ select · Enter switch · Esc back · q close</OverlayHint>
      </Box>
    )
  }

  // ── Model selection stage ────────────────────────────────────────────
  const { items, offset } = windowItems(models, modelIdx, VISIBLE)
  const noModelMatches = !!filter.trim() && models.length === 0

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        Select model (step 2/3)
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {filteredProviderRows[providerIdx]?.name || '(unknown provider)'} · Esc back
      </Text>
      <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
        {filter ? `filter: ${filter}▎` : 'type to filter · ↑/↓ select'}
      </Text>
      <Text color={t.color.label} wrap="truncate-end">
        {provider?.warning ? `warning: ${provider.warning}` : ' '}
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {offset > 0 ? ` ↑ ${offset} more` : ' '}
      </Text>

      {Array.from({ length: VISIBLE }, (_, i) => {
        const row = items[i]
        const idx = offset + i

        if (!row) {
          return (!allModels.length || noModelMatches) && i === 0 ? (
            <Text color={t.color.muted} key="empty" wrap="truncate-end">
              {noModelMatches ? 'no models match filter' : 'no models listed for this provider'}
            </Text>
          ) : (
            <Text color={t.color.muted} key={`pad-${i}`} wrap="truncate-end">
              {' '}
            </Text>
          )
        }

        const prefix = modelIdx === idx ? '▸ ' : row === currentModel ? '* ' : '  '

        return (
          <Text
            color={t.color.muted}
            {...chipRowProps(t, modelIdx === idx)}
            key={`${provider?.slug ?? 'prov'}:${idx}:${row}`}
            wrap="truncate-end"
          >
            {prefix}
            {idx + 1}. <HitLabel q={filter} t={t} text={row} />
          </Text>
        )
      })}

      <Text color={t.color.muted} wrap="truncate-end">
        {offset + VISIBLE < models.length ? ` ↓ ${models.length - offset - VISIBLE} more` : ' '}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        persist: {allowPersistGlobal ? (persistGlobal ? 'global' : 'session') : 'session'}
        {allowPersistGlobal ? ' · ^g toggle' : ' only'}
      </Text>
      <OverlayHint t={t}>
        {models.length ? '↑/↓ select · Enter next · Esc clear/back · q close' : 'Esc back · q close'}
      </OverlayHint>
    </Box>
  )
}

interface ModelPickerProps {
  allowPersistGlobal?: boolean
  gw: GatewayClient
  initialRefresh?: boolean
  initialStage?: 'hop' | 'provider'
  maxWidth?: number
  onCancel: () => void
  onSelect: (value: string) => void
  sessionId: string | null
  t: Theme
}
