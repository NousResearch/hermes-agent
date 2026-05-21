import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { useI18n } from '../i18n/index.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

import { OverlayHint, useOverlayKeys, windowItems, windowOffset } from './overlayControls.js'

const VISIBLE = 12
const MIN_WIDTH = 40
const MAX_WIDTH = 90

export function SkillsHub({ gw, onClose, t }: SkillsHubProps) {
  const { t: ti } = useI18n()
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
    return <Text color={t.color.muted}>{ti('skills.loading')}</Text>
  }

  if (err && stage === 'category') {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.label}>{ti('sys.error', { message: err })}</Text>
        <OverlayHint t={t}>{ti('picker.cancel')}</OverlayHint>
      </Box>
    )
  }

  if (!cats.length) {
    return (
      <Box flexDirection="column" width={width}>
        <Text color={t.color.muted}>{ti('sys.noSkills')}</Text>
        <OverlayHint t={t}>{ti('picker.cancel')}</OverlayHint>
      </Box>
    )
  }

  if (stage === 'category') {
    const rows = cats.map(c => ti('skills.categoryCount', { category: c, count: String(skillsByCat[c]?.length ?? 0) }))
    const { items, offset } = windowItems(rows, catIdx, VISIBLE)

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent}>
          {ti('skills.hubTitle')}
        </Text>

        <Text color={t.color.muted}>{ti('skills.selectCategory')}</Text>
        {offset > 0 && <Text color={t.color.muted}> {ti('sys.moreAbove', { count: String(offset) })}</Text>}

        {items.map((row, i) => {
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
        })}

        {offset + VISIBLE < rows.length && <Text color={t.color.muted}> {ti('sys.moreBelow', { count: String(rows.length - offset - VISIBLE) })}</Text>}
        <OverlayHint t={t}>{ti('picker.skillHint')}</OverlayHint>
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

        <Text color={t.color.muted}>{ti('skills.count', { count: String(skills.length) })}</Text>
        {!skills.length ? <Text color={t.color.muted}>{ti('skills.noneInCategory')}</Text> : null}
        {offset > 0 && <Text color={t.color.muted}> {ti('sys.moreAbove', { count: String(offset) })}</Text>}

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

        {offset + VISIBLE < skills.length && (
          <Text color={t.color.muted}> {ti('sys.moreBelow', { count: String(skills.length - offset - VISIBLE) })}</Text>
        )}
        <OverlayHint t={t}>
          {skills.length ? ti('skills.listHint') : ti('skills.backCloseHint')}
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
      {info?.path ? <Text color={t.color.muted}>{ti('skills.path', { path: info.path })}</Text> : null}
      {!info && !err ? <Text color={t.color.muted}>{ti('sys.loading')}</Text> : null}
      {err ? <Text color={t.color.label}>{ti('sys.error', { message: err })}</Text> : null}
      {installing ? <Text color={t.color.accent}>{ti('skills.installing')}</Text> : null}

      <OverlayHint t={t}>{ti('skills.actionsHint')}</OverlayHint>
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
