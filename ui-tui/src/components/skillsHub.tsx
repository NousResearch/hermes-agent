import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { fuzzyRank } from '../lib/fuzzy.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { clampIndex, OverlayHint, useOverlayKeys, windowItems, windowOffset } from './overlayControls.js'

const VISIBLE = 12
const MIN_WIDTH = 40
const MAX_WIDTH = 90

// ── Type-to-filter helpers ───────────────────────────────────────────
//
// The category + skill list stages support fuzzy type-to-filter, mirroring
// modelPicker. The list logic and the key-gating decision are factored into
// pure functions so they can be unit-tested without rendering Ink.

/** Subset of the Ink key flags the list-stage handler inspects. */
export interface SkillsKeyFlags {
  backspace?: boolean
  ctrl?: boolean
  delete?: boolean
  downArrow?: boolean
  escape?: boolean
  meta?: boolean
  return?: boolean
  upArrow?: boolean
}

/**
 * Rank skill categories by a fuzzy query against the category name plus the
 * names of the skills it contains, so typing a skill name surfaces its
 * category. An empty query keeps the original (sorted) order.
 */
export const rankSkillCategories = (
  cats: readonly string[],
  skillsByCat: Record<string, string[]>,
  query: string
): string[] =>
  query.trim() ? fuzzyRank(cats, query, c => `${c} ${(skillsByCat[c] ?? []).join(' ')}`).map(r => r.item) : [...cats]

/** Rank skills within a category by a fuzzy query. Empty query keeps order. */
export const rankSkills = (skills: readonly string[], query: string): string[] =>
  query.trim() ? fuzzyRank(skills, query, s => s).map(r => r.item) : [...skills]

export type SkillsListAction =
  | { ch: string; kind: 'append' }
  | { kind: 'backspace' }
  | { kind: 'clearFilter' }
  | { kind: 'close' }
  | { kind: 'down' }
  | { kind: 'escape' }
  | { kind: 'ignore' }
  | { kind: 'quick'; n: number }
  | { kind: 'select' }
  | { kind: 'up' }

/**
 * Decide what a keypress does on a list stage. Reserved single-key shortcuts
 * (`q` to close, `1`-`9`/`0` quick-select) only fire when no filter is active,
 * so the same keys extend the filter once the user is typing one. Navigation,
 * Backspace, Ctrl+U and Esc are always available. Any other printable key
 * starts/extends the filter.
 */
export function classifySkillsListKey(ch: string, key: SkillsKeyFlags, hasFilter: boolean): SkillsListAction {
  if (key.escape) {
    return hasFilter ? { kind: 'clearFilter' } : { kind: 'escape' }
  }

  if (key.upArrow) {
    return { kind: 'up' }
  }

  if (key.downArrow) {
    return { kind: 'down' }
  }

  if (key.return) {
    return { kind: 'select' }
  }

  if (key.backspace || key.delete) {
    return hasFilter ? { kind: 'backspace' } : { kind: 'ignore' }
  }

  if (key.ctrl && ch === 'u') {
    return hasFilter ? { kind: 'clearFilter' } : { kind: 'ignore' }
  }

  // Reserved shortcuts fire only when not actively filtering.
  if (!hasFilter) {
    if (ch === 'q' && !key.ctrl && !key.meta) {
      return { kind: 'close' }
    }

    const n = ch === '0' ? 10 : Number.parseInt(ch, 10)

    if (!Number.isNaN(n) && n >= 1 && n <= 10) {
      return { kind: 'quick', n }
    }
  }

  if (ch && !key.ctrl && !key.meta && ch.length === 1 && ch >= ' ') {
    // A leading space must not start a filter: it would read as "active" (and
    // disable the reserved shortcuts) while the trimmed query leaves the list
    // unfiltered. Spaces inside an existing query are fine (multi-token match).
    if (ch === ' ' && !hasFilter) {
      return { kind: 'ignore' }
    }

    return { ch, kind: 'append' }
  }

  return { kind: 'ignore' }
}

export function SkillsHub({ gw, onClose, t }: SkillsHubProps) {
  const [skillsByCat, setSkillsByCat] = useState<Record<string, string[]>>({})
  const [selectedCat, setSelectedCat] = useState('')
  const [catIdx, setCatIdx] = useState(0)
  const [skillIdx, setSkillIdx] = useState(0)
  const [stage, setStage] = useState<'actions' | 'category' | 'skill'>('category')
  const [info, setInfo] = useState<null | SkillInfo>(null)
  const [installing, setInstalling] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  // Type-to-filter query, scoped per list stage (cleared on stage change).
  const [filter, setFilter] = useState('')

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  useEffect(() => {
    gw.request<{ skills?: Record<string, string[]> }>('skills.manage', { action: 'list' })
      .then(r => {
        setSkillsByCat(r?.skills ?? {})
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }, [gw])

  // Memoized so the filtered useMemos below have stable inputs (otherwise the
  // fuzzy rank would re-run every render).
  const cats = useMemo(() => Object.keys(skillsByCat).sort(), [skillsByCat])
  const skills = useMemo(() => (selectedCat ? (skillsByCat[selectedCat] ?? []) : []), [selectedCat, skillsByCat])
  const skillName = skills[skillIdx] ?? ''

  // Filtered views drive navigation + rendering; indices point into these.
  const filteredCats = useMemo(() => rankSkillCategories(cats, skillsByCat, filter), [cats, skillsByCat, filter])
  const filteredSkills = useMemo(() => rankSkills(skills, filter), [skills, filter])

  const listStage = stage === 'category' || stage === 'skill'

  // Keep the active selection within the (possibly shrunk) filtered list.
  useEffect(() => {
    setCatIdx(i => clampIndex(i, filteredCats.length))
  }, [filteredCats.length])

  useEffect(() => {
    setSkillIdx(i => clampIndex(i, filteredSkills.length))
  }, [filteredSkills.length])

  const back = () => {
    if (stage === 'actions') {
      setStage('skill')
      setInfo(null)
      setErr('')

      return
    }

    if (stage === 'skill') {
      setStage('category')
      setSkillIdx(0)
      setFilter('')

      return
    }

    onClose()
  }

  // Disable the shared overlay q/Esc handler on the list stages so those keys
  // can be typed into / used to clear the filter (handled locally below).
  useOverlayKeys({ disabled: installing || listStage, onBack: back, onClose })

  const inspect = (name: string) => {
    setInfo(null)
    setErr('')

    gw.request<{ info?: SkillInfo }>('skills.manage', { action: 'inspect', query: name })
      .then(r => setInfo(r?.info ?? { name }))
      .catch((e: unknown) => setErr(rpcErrorMessage(e)))
  }

  const install = (name: string) => {
    setInstalling(true)
    setErr('')

    gw.request<{ installed?: boolean; name?: string }>('skills.manage', { action: 'install', query: name })
      .then(() => onClose())
      .catch((e: unknown) => setErr(rpcErrorMessage(e)))
      .finally(() => setInstalling(false))
  }

  // Open the category at `idx` in the filtered list, re-anchoring catIdx to the
  // full (sorted) list so it stays valid once the filter clears.
  const openCategory = (idx: number) => {
    const cat = filteredCats[idx]

    if (!cat) {
      return
    }

    setSelectedCat(cat)
    setCatIdx(Math.max(0, cats.indexOf(cat)))
    setSkillIdx(0)
    setStage('skill')
    setFilter('')
  }

  // Open the skill at `idx` in the filtered list, re-anchoring skillIdx to the
  // full category list so the actions stage reads the right skill name.
  const openSkill = (idx: number) => {
    const name = filteredSkills[idx]

    if (!name) {
      return
    }

    setSkillIdx(Math.max(0, skills.indexOf(name)))
    setStage('actions')
    setFilter('')
    inspect(name)
  }

  useInput((ch, key) => {
    if (installing) {
      return
    }

    if (stage === 'actions') {
      if (key.return) {
        setStage('skill')
        setInfo(null)
        setErr('')

        return
      }

      if (ch.toLowerCase() === 'x' && skillName) {
        install(skillName)

        return
      }

      if (ch.toLowerCase() === 'i' && skillName) {
        inspect(skillName)
      }

      return
    }

    const list = stage === 'category' ? filteredCats : filteredSkills
    const count = list.length
    const sel = stage === 'category' ? catIdx : skillIdx
    const setSel = stage === 'category' ? setCatIdx : setSkillIdx
    const action = classifySkillsListKey(ch, key, filter.trim() !== '')

    switch (action.kind) {
      case 'append':
        setFilter(v => v + action.ch)
        setSel(0)

        return

      case 'backspace':
        setFilter(v => v.slice(0, -1))
        setSel(0)

        return

      case 'clearFilter':
        setFilter('')
        setSel(0)

        return

      case 'close':
        return onClose()

      case 'down':
        if (sel < count - 1) {
          setSel(v => v + 1)
        }

        return

      case 'escape':
        return back()

      case 'select':
        if (stage === 'category') {
          openCategory(catIdx)
        } else {
          openSkill(skillIdx)
        }

        return

      case 'up':
        if (sel > 0) {
          setSel(v => v - 1)
        }

        return
      case 'quick': {
        if (action.n <= Math.min(10, count)) {
          const next = windowOffset(count, sel, VISIBLE) + action.n - 1

          if (stage === 'category') {
            openCategory(next)
          } else {
            openSkill(next)
          }
        }

        return
      }

      default:
        return
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading skills…</Text>
  }

  if (err && stage === 'category') {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (!cats.length) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.muted}>no skills available</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (stage === 'category') {
    const rows = filteredCats.map(c => `${c} · ${skillsByCat[c]?.length ?? 0} skills`)
    const { items, offset } = windowItems(rows, catIdx, VISIBLE)
    const noMatches = filter.trim() !== '' && rows.length === 0

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          Skills Hub
        </Text>

        <Text color={t.color.muted}>select a category</Text>
        <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
          {filter ? `filter: ${filter}▎` : 'type to filter · ↑/↓ select'}
        </Text>
        {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

        {noMatches ? (
          <Text color={t.color.muted}>no categories match</Text>
        ) : (
          items.map((row, i) => {
            const idx = offset + i

            return (
              <Text
                bold={catIdx === idx}
                color={catIdx === idx ? t.color.accent : t.color.muted}
                inverse={catIdx === idx}
                key={row}
                wrap="truncate-end"
              >
                {catIdx === idx ? '▸ ' : '  '}
                {i + 1}. {row}
              </Text>
            )
          })
        )}

        {offset + VISIBLE < rows.length && <Text color={t.color.muted}> ↓ {rows.length - offset - VISIBLE} more</Text>}
        <OverlayHint t={t}>
          {filter.trim()
            ? filteredCats.length
              ? '↑/↓ select · Enter open · Backspace edit · Esc clear filter'
              : 'Backspace edit · Esc clear filter'
            : '↑/↓ select · Enter open · 1-9,0 quick · type to filter · q close'}
        </OverlayHint>
      </Box>
    )
  }

  if (stage === 'skill') {
    const { items, offset } = windowItems(filteredSkills, skillIdx, VISIBLE)
    // Only a non-empty category can show "no matches"; an empty category keeps
    // its own "no skills in this category" message (avoid showing both).
    const noMatches = filter.trim() !== '' && skills.length > 0 && filteredSkills.length === 0

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          {selectedCat}
        </Text>

        <Text color={t.color.muted}>{skills.length} skill(s)</Text>
        <Text color={filter ? t.color.accent : t.color.muted} wrap="truncate-end">
          {filter ? `filter: ${filter}▎` : 'type to filter · ↑/↓ select'}
        </Text>
        {!skills.length ? <Text color={t.color.muted}>no skills in this category</Text> : null}
        {noMatches ? <Text color={t.color.muted}>no skills match</Text> : null}
        {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

        {items.map((row, i) => {
          const idx = offset + i

          return (
            <Text
              bold={skillIdx === idx}
              color={skillIdx === idx ? t.color.accent : t.color.muted}
              inverse={skillIdx === idx}
              key={row}
              wrap="truncate-end"
            >
              {skillIdx === idx ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          )
        })}

        {offset + VISIBLE < filteredSkills.length && (
          <Text color={t.color.muted}> ↓ {filteredSkills.length - offset - VISIBLE} more</Text>
        )}
        <OverlayHint t={t}>
          {filter.trim()
            ? filteredSkills.length
              ? '↑/↓ select · Enter open · Backspace edit · Esc clear filter'
              : 'Backspace edit · Esc clear filter'
            : filteredSkills.length
              ? '↑/↓ select · Enter open · 1-9,0 quick · type to filter · Esc back · q close'
              : 'Esc back · q close'}
        </OverlayHint>
      </Box>
    )
  }

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        {info?.name ?? skillName}
      </Text>

      <Text color={t.color.muted}>{info?.category ?? selectedCat}</Text>
      {info?.description ? <Text color={t.color.text}>{info.description}</Text> : null}
      {info?.path ? <Text color={t.color.muted}>path: {info.path}</Text> : null}
      {!info && !err ? <Text color={t.color.muted}>loading…</Text> : null}
      {err ? <Text color={t.color.label}>error: {err}</Text> : null}
      {installing ? <Text color={t.color.accent}>installing…</Text> : null}

      <OverlayHint t={t}>i reinspect · x reinstall · Enter/Esc back · q close</OverlayHint>
    </Box>
  )
}

interface SkillInfo {
  category?: string
  description?: string
  name?: string
  path?: string
}

interface SkillsHubProps {
  gw: GatewayClient
  onClose: () => void
  t: Theme
}
