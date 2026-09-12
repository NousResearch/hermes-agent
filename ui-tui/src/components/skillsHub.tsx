import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { useTranslations } from '../i18n/index.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, useOverlayKeys, windowItems, windowOffset } from './overlayControls.js'
import { chipRowProps } from './overlayPrimitives.js'
import { clampOverlayWidth } from './overlayPrimitives.js'

const VISIBLE = 12
const MIN_WIDTH = 40
const MAX_WIDTH = 90

export function SkillsHub({ gw, maxWidth, onClose, t }: SkillsHubProps) {
  const copy = useTranslations()
  const [skillsByCat, setSkillsByCat] = useState<Record<string, string[]>>({})
  const [selectedCat, setSelectedCat] = useState('')
  const [catIdx, setCatIdx] = useState(0)
  const [skillIdx, setSkillIdx] = useState(0)
  const [stage, setStage] = useState<'actions' | 'category' | 'skill'>('category')
  const [info, setInfo] = useState<null | SkillInfo>(null)
  const [installing, setInstalling] = useState(false)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)

  const { stdout } = useStdout()
  const terminalWidth = Math.max(1, (stdout?.columns ?? 80) - 6)
  const preferredWidth = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, terminalWidth))
  const width = clampOverlayWidth(preferredWidth, maxWidth)

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

  const cats = Object.keys(skillsByCat).sort()
  const skills = selectedCat ? (skillsByCat[selectedCat] ?? []) : []
  const skillName = skills[skillIdx] ?? ''

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

      return
    }

    onClose()
  }

  useOverlayKeys({ disabled: installing, onBack: back, onClose })

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

    const count = stage === 'category' ? cats.length : skills.length
    const sel = stage === 'category' ? catIdx : skillIdx
    const setSel = stage === 'category' ? setCatIdx : setSkillIdx

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)

      return
    }

    if (key.downArrow && sel < count - 1) {
      setSel(v => v + 1)

      return
    }

    if (key.return) {
      if (stage === 'category') {
        const cat = cats[catIdx]

        if (!cat) {
          return
        }

        setSelectedCat(cat)
        setSkillIdx(0)
        setStage('skill')

        return
      }

      const name = skills[skillIdx]

      if (name) {
        setStage('actions')
        inspect(name)
      }

      return
    }

    const n = ch === '0' ? 10 : parseInt(ch, 10)

    if (!Number.isNaN(n) && n >= 1 && n <= Math.min(10, count)) {
      const next = windowOffset(count, sel, VISIBLE) + n - 1

      if (stage === 'category') {
        const cat = cats[next]

        if (cat) {
          setSelectedCat(cat)
          setCatIdx(next)
          setSkillIdx(0)
          setStage('skill')
        }

        return
      }

      const name = skills[next]

      if (name) {
        setSkillIdx(next)
        setStage('actions')
        inspect(name)
      }
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>{copy.skills.loading}</Text>
  }

  if (err && stage === 'category') {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>
          {copy.panels.error}
          {err}
        </Text>
        <OverlayHint t={t}>{copy.panels.cancel}</OverlayHint>
      </Box>
    )
  }

  if (!cats.length) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.muted}>{copy.skills.none}</Text>
        <OverlayHint t={t}>{copy.panels.cancel}</OverlayHint>
      </Box>
    )
  }

  if (stage === 'category') {
    const rows = cats.map(c => copy.skills.categoryCount(c, skillsByCat[c]?.length ?? 0))
    const { items, offset } = windowItems(rows, catIdx, VISIBLE)

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          {copy.skills.title}
        </Text>

        <Text color={t.color.muted}>{copy.skills.category}</Text>
        {offset > 0 && <Text color={t.color.muted}>{copy.panels.moreAbove(offset)}</Text>}

        {items.map((row, i) => {
          const idx = offset + i

          return (
            <Text color={t.color.muted} {...chipRowProps(t, catIdx === idx)} key={row} wrap="truncate-end">
              {catIdx === idx ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          )
        })}

        {offset + VISIBLE < rows.length && (
          <Text color={t.color.muted}>{copy.panels.moreBelow(rows.length - offset - VISIBLE)}</Text>
        )}
        <OverlayHint t={t}>{copy.panels.openHint}</OverlayHint>
      </Box>
    )
  }

  if (stage === 'skill') {
    const { items, offset } = windowItems(skills, skillIdx, VISIBLE)

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          {selectedCat}
        </Text>

        <Text color={t.color.muted}>{copy.skills.count(skills.length)}</Text>
        {!skills.length ? <Text color={t.color.muted}>{copy.skills.emptyCategory}</Text> : null}
        {offset > 0 && <Text color={t.color.muted}>{copy.panels.moreAbove(offset)}</Text>}

        {items.map((row, i) => {
          const idx = offset + i

          return (
            <Text color={t.color.muted} {...chipRowProps(t, skillIdx === idx)} key={row} wrap="truncate-end">
              {skillIdx === idx ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          )
        })}

        {offset + VISIBLE < skills.length && (
          <Text color={t.color.muted}>{copy.panels.moreBelow(skills.length - offset - VISIBLE)}</Text>
        )}
        <OverlayHint t={t}>{skills.length ? copy.panels.openBackHint : copy.panels.back}</OverlayHint>
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
      {info?.path ? (
        <Text color={t.color.muted}>
          {copy.panels.path}
          {info.path}
        </Text>
      ) : null}
      {!info && !err ? <Text color={t.color.muted}>{copy.panels.loading}</Text> : null}
      {err ? (
        <Text color={t.color.label}>
          {copy.panels.error}
          {err}
        </Text>
      ) : null}
      {installing ? <Text color={t.color.accent}>{copy.skills.installing}</Text> : null}

      <OverlayHint t={t}>{copy.skills.inspectHint}</OverlayHint>
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
  maxWidth?: number
  onClose: () => void
  t: Theme
}
