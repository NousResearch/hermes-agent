import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { fuzzyRank, searchFold } from '@hermes/shared/fuzzy'
import type { ModelOptionProvider, ModelOptionsResult } from '@hermes/shared/gateway-events'
import { modelSearchText } from '@hermes/shared/model-search-text'
import { REASONING_EFFORTS } from '@hermes/shared/reasoning-effort'
import { useEffect, useMemo, useState } from 'react'

import { providerDisplayNames } from '../domain/providers.js'
import { TUI_SESSION_MODEL_FLAG } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import { messages } from '../i18n/runtime.js'
import { useT } from '../i18n/useT.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, useOverlayKeys, windowItems } from './overlayControls.js'
import { chipRowProps, clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 12
const MIN_WIDTH = 40
const MAX_WIDTH = 90

type Stage = 'provider' | 'key' | 'model' | 'reasoning' | 'disconnect'

type ProviderRow = { name: string; provider: ModelOptionProvider }

/** Rows of the effort step (step 3/3): the shared ladder, the off state, then
 *  "keep current" (empty value = no `--reasoning` flag on the emitted command).
 *  Resolved at call time so the two labelled rows follow the active language;
 *  the ladder levels are the literal `--reasoning` values and stay as-is. */
export function reasoningPickerRows(): ReadonlyArray<{ label: string; value: string }> {
  const r = messages().pickers.model.reasoning

  return [
    ...REASONING_EFFORTS.map(level => ({ label: level, value: level })),
    { label: r.none, value: 'none' },
    { label: r.keepCurrent, value: '' }
  ]
}

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

type ProviderStageRow = ({ kind: 'provider' } | { kind: 'model'; model: string }) & ProviderRow

/** Step-1 rows. With a query, each authenticated provider's models rank alongside the provider
 *  rows (matched on their own id, like step 2), so a typed model is one Enter away; a provider
 *  whose name or slug contains the query stays on top. An empty query is the plain provider list. */
export function providerStageRows(providerRows: ProviderRow[], filter: string): ProviderStageRow[] {
  const rows: ProviderStageRow[] = providerRows.map(row => ({ kind: 'provider' as const, ...row }))
  const query = searchFold(filter.trim())

  if (!query) {
    return rows
  }

  const named = rows.filter(row => searchFold(`${row.name} ${row.provider.slug}`).includes(query))

  // Current provider first, so an id several providers serve stays on the current route on a score tie.
  const byCurrent = [...providerRows].sort((a, b) => Number(!!b.provider.is_current) - Number(!!a.provider.is_current))

  const models = byCurrent.flatMap(({ name, provider }) =>
    provider.authenticated === false
      ? []
      : (provider.models ?? []).map(model => ({ kind: 'model' as const, model, name, provider }))
  )

  const ranked = fuzzyRank([...rows.filter(row => !named.includes(row)), ...models], filter, row =>
    row.kind === 'model'
      ? modelSearchText(row.model)
      : `${row.name} ${row.provider.slug} ${(row.provider.models ?? []).join(' ')}`
  )

  return [...named, ...ranked.map(r => r.item)]
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

export function ModelPicker({
  allowPersistGlobal = true,
  gw,
  initialRefresh = false,
  maxWidth,
  onCancel,
  onSelect,
  sessionId,
  t
}: ModelPickerProps) {
  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [currentModel, setCurrentModel] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const [persistGlobal, setPersistGlobal] = useState(false)
  const [providerIdx, setProviderIdx] = useState(0)
  const [modelIdx, setModelIdx] = useState(0)
  const [reasoningIdx, setReasoningIdx] = useState(0)
  // Model chosen on step 2, awaiting the effort pick on step 3.
  const [pendingModel, setPendingModel] = useState('')
  const [stage, setStage] = useState<Stage>('provider')
  const [keyInput, setKeyInput] = useState('')
  const [keySaving, setKeySaving] = useState(false)
  const [keyError, setKeyError] = useState('')
  // Type-to-filter query, scoped per stage (cleared on stage change).
  const [filter, setFilter] = useState('')
  const T = useT()
  const M = T.pickers.model
  const C = T.pickers.common
  const reasoningRows = reasoningPickerRows()

  const { stdout } = useStdout()
  // Pin the picker to a stable width so the FloatBox parent (which shrinks-
  // to-fit with alignSelf="flex-start") doesn't resize as long provider /
  // model names scroll into view, and so `wrap="truncate-end"` on each row
  // has an actual constraint to truncate against. Optional maxWidth lets
  // grid layouts hand the picker its cell budget.
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

  // "persist: global · ^g toggle" / "persist: session only" footer shared by the list stages.
  const persistLine =
    M.persist.label(allowPersistGlobal && persistGlobal ? M.persist.scopeGlobal : M.persist.scopeSession) +
    (allowPersistGlobal ? M.persist.toggleHint : M.persist.onlySuffix)

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
          setErr(messages().pickers.common.invalidResponse('model.options'))
          setLoading(false)

          return
        }

        const next = r.providers ?? []
        setProviders(next)
        setCurrentModel(String(r.model ?? ''))
        setProviderIdx(
          Math.max(
            0,
            next.findIndex(p => p.is_current)
          )
        )
        setModelIdx(0)
        setStage('provider')
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

  // providerIdx / modelIdx always index into the *displayed* (filtered) lists;
  // a step-1 query can interleave model rows (see providerStageRows).
  // With an empty filter the filtered list equals the full list, so navigation
  // behaves exactly as before. Filtering only applies on the relevant stage.
  const filteredProviderRows = useMemo(
    () => providerStageRows(providerRows, stage === 'provider' ? filter : ''),
    [providerRows, filter, stage]
  )

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

  const back = () => {
    // Esc first clears an active filter on the list stages, before navigating.
    if ((stage === 'provider' || stage === 'model') && filter.trim()) {
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
      setStage('model')
      setPendingModel('')
      setReasoningIdx(0)

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
  const listStage = stage === 'provider' || stage === 'model' || stage === 'reasoning'
  useOverlayKeys({ disabled: listStage, onBack: back, onClose: onCancel })

  useInput((ch, key) => {
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
              setKeyError(M.failedToSaveKey)
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
                        warning: p.key_env ? M.pasteKeyToActivate(p.key_env) : M.runHermesModelToConfigure
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

      if (key.downArrow && reasoningIdx < reasoningRows.length - 1) {
        setReasoningIdx(v => v + 1)

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
            reasoningRows[reasoningIdx]?.value ?? ''
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

    const count = stage === 'provider' ? filteredProviderRows.length : models.length
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

    if (key.return) {
      if (stage === 'provider') {
        const row = filteredProviderRows[providerIdx]

        // A model hit switches directly, like step 2's "keep current effort" pick (no --reasoning).
        if (row?.kind === 'model') {
          onSelect(modelPickerCommand(row.model, row.provider.slug, allowPersistGlobal && persistGlobal))

          return
        }

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
        if (pickerOffersReasoning(provider, model)) {
          // Step 3/3: effort for the picked model (skipped on reasoning-free routes).
          setPendingModel(model)
          setReasoningIdx(0)
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

    // Any other printable single character extends the filter — once the list has loaded, so the
    // selection restored on load can't point at an arbitrary row of an already-filtered list.
    if (!loading && ch && !key.ctrl && !key.meta && ch.length === 1 && ch >= ' ') {
      setFilter(v => v + ch)
      setSel(0)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>{M.loading}</Text>
  }

  if (err) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>{C.error(err)}</Text>
        <OverlayHint t={t}>{M.escQCancelHint}</OverlayHint>
      </Box>
    )
  }

  if (!providers.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>{M.noProviders}</Text>
        <OverlayHint t={t}>{M.escQCancelHint}</OverlayHint>
      </Box>
    )
  }

  // ── Key entry stage ──────────────────────────────────────────────────
  if (stage === 'key' && provider) {
    const masked = keyInput ? '•'.repeat(Math.min(keyInput.length, 40)) : ''

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          {M.key.title(provider.name)}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {M.key.pasteBelow}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {provider.key_env}:
        </Text>

        <Text color={t.color.accent} wrap="truncate-end">
          {'  '}
          {masked || M.key.empty}
          {keySaving ? '' : '▎'}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        {keyError ? (
          <Text color={t.color.label} wrap="truncate-end">
            {C.error(keyError)}
          </Text>
        ) : keySaving ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {M.key.saving}
          </Text>
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {' '}
          </Text>
        )}

        <OverlayHint t={t}>{M.key.hint}</OverlayHint>
      </Box>
    )
  }

  // ── Disconnect confirmation stage ─────────────────────────────────────
  if (stage === 'disconnect' && provider) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          {M.disconnect.title(provider.name)}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {M.disconnect.removesCredentials(provider.name)}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {M.disconnect.reauthLater}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        {keySaving ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {M.disconnect.disconnecting}
          </Text>
        ) : (
          <OverlayHint t={t}>{M.disconnect.hint}</OverlayHint>
        )}
      </Box>
    )
  }

  // ── Provider selection stage ─────────────────────────────────────────
  if (stage === 'provider') {
    const rows = filteredProviderRows.map(row => {
      if (row.kind === 'model') {
        return `${row.provider.is_current && row.model === currentModel ? '*' : ' '} ${row.name} · ${row.model}`
      }

      const { provider: p, name } = row
      const authMark = p.authenticated === false ? '○' : p.is_current ? '*' : '●'
      const modelCount = p.total_models ?? p.models?.length ?? 0

      const suffix =
        p.authenticated === false
          ? p.auth_type === 'api_key'
            ? M.provider.noKey
            : M.provider.needsSetup
          : modelCount === 1
            ? M.provider.modelCountOne(1)
            : M.provider.modelCountOther(modelCount)

      return `${authMark} ${name} · ${suffix}`
    })

    const { items, offset } = windowItems(rows, providerIdx, VISIBLE)
    const noMatches = !!filter.trim() && rows.length === 0

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          {M.provider.title}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {M.provider.subtitle}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {M.provider.current(currentModel || M.provider.unknown)}
        </Text>
        <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
          {filter ? `${C.filter(filter)}▎` : M.provider.filterHint}
        </Text>
        <Text color={t.color.label} wrap="truncate-end">
          {provider?.warning ? M.provider.warning(provider.warning) : ' '}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          {offset > 0 ? C.moreAbove(offset) : ' '}
        </Text>

        {noMatches ? (
          <Text color={t.color.muted} wrap="truncate-end">
            {M.provider.noMatches}
          </Text>
        ) : (
          Array.from({ length: VISIBLE }, (_, i) => {
            const row = items[i]
            const idx = offset + i
            const entry = filteredProviderRows[idx]
            const p = entry?.provider
            const dimmed = p?.authenticated === false

            return row ? (
              <Text
                color={dimmed ? t.color.label : t.color.muted}
                {...chipRowProps(t, providerIdx === idx)}
                key={entry?.kind === 'model' ? `${p?.slug}:${idx}:${entry.model}` : (p?.slug ?? `row-${idx}`)}
                wrap="truncate-end"
              >
                {providerIdx === idx ? '▸ ' : '  '}
                {idx + 1}. {row}
              </Text>
            ) : (
              <Text color={t.color.muted} key={`pad-${i}`} wrap="truncate-end">
                {' '}
              </Text>
            )
          })
        )}

        <Text color={t.color.muted} wrap="truncate-end">
          {offset + VISIBLE < rows.length ? C.moreBelow(rows.length - offset - VISIBLE) : ' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {persistLine}
        </Text>
        <OverlayHint t={t}>{M.provider.hint}</OverlayHint>
      </Box>
    )
  }

  // ── Reasoning effort stage ───────────────────────────────────────────
  if (stage === 'reasoning') {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          {M.effort.title}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {M.effort.subtitle(pendingModel)}
        </Text>

        {reasoningRows.map((row, idx) => (
          <Text
            color={t.color.muted}
            {...chipRowProps(t, reasoningIdx === idx)}
            key={row.value || 'keep'}
            wrap="truncate-end"
          >
            {reasoningIdx === idx ? '▸ ' : '  '}
            {idx + 1}. {row.label}
          </Text>
        ))}

        <Text color={t.color.muted} wrap="truncate-end">
          {persistLine}
        </Text>
        <OverlayHint t={t}>{M.effort.hint}</OverlayHint>
      </Box>
    )
  }

  // ── Model selection stage ────────────────────────────────────────────
  const { items, offset } = windowItems(models, modelIdx, VISIBLE)
  const noModelMatches = !!filter.trim() && models.length === 0

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        {M.modelStage.title}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {M.modelStage.subtitle(filteredProviderRows[providerIdx]?.name || M.modelStage.unknownProvider)}
      </Text>
      <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
        {filter ? `${C.filter(filter)}▎` : M.provider.filterHint}
      </Text>
      <Text color={t.color.label} wrap="truncate-end">
        {provider?.warning ? M.provider.warning(provider.warning) : ' '}
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {offset > 0 ? C.moreAbove(offset) : ' '}
      </Text>

      {Array.from({ length: VISIBLE }, (_, i) => {
        const row = items[i]
        const idx = offset + i

        if (!row) {
          return (!allModels.length || noModelMatches) && i === 0 ? (
            <Text color={t.color.muted} key="empty" wrap="truncate-end">
              {noModelMatches ? M.modelStage.noMatches : M.modelStage.noneListed}
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
            {idx + 1}. {row}
          </Text>
        )
      })}

      <Text color={t.color.muted} wrap="truncate-end">
        {offset + VISIBLE < models.length ? C.moreBelow(models.length - offset - VISIBLE) : ' '}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {persistLine}
      </Text>
      <OverlayHint t={t}>{models.length ? M.modelStage.hint : M.modelStage.emptyHint}</OverlayHint>
    </Box>
  )
}

interface ModelPickerProps {
  allowPersistGlobal?: boolean
  gw: GatewayClient
  initialRefresh?: boolean
  maxWidth?: number
  onCancel: () => void
  onSelect: (value: string) => void
  sessionId: string | null
  t: Theme
}
