import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { GatewayClient } from '../gatewayClient.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import type { Theme } from '../theme.js'

const VISIBLE = 12

const pageOffset = (count: number, sel: number) => Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), count - VISIBLE))

const visibleItems = (items: string[], sel: number) => {
  const off = pageOffset(items.length, sel)

  return { items: items.slice(off, off + VISIBLE), off }
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

    if (key.escape) {
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

      return
    }

    if (stage === 'actions') {
      if (key.return || ch.toLowerCase() === 'x') {
        if (skillName) {
          install(skillName)
        }

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
      const off = pageOffset(count, sel)
      const next = off + n - 1

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
    return <Text color={t.color.dim}>loading skills…</Text>
  }

  if (err && stage === 'category') {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <Text color={t.color.dim}>Esc to cancel</Text>
      </Box>
    )
  }

  if (!cats.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.dim}>no skills available</Text>
        <Text color={t.color.dim}>Esc to cancel</Text>
      </Box>
    )
  }

  if (stage === 'category') {
    const rows = cats.map(c => `${c} · ${skillsByCat[c]?.length ?? 0} skills`)
    const { items, off } = visibleItems(rows, catIdx)

    return (
      <Box flexDirection="column">
        <Text bold color={t.color.amber}>
          Skills Hub
        </Text>

        <Text color={t.color.dim}>select a category</Text>
        {off > 0 && <Text color={t.color.dim}> ↑ {off} more</Text>}

        {items.map((row, i) => {
          const idx = off + i

          return (
            <Text color={catIdx === idx ? t.color.cornsilk : t.color.dim} key={row}>
              {catIdx === idx ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          )
        })}

        {off + VISIBLE < rows.length && <Text color={t.color.dim}> ↓ {rows.length - off - VISIBLE} more</Text>}
        <Text color={t.color.dim}>↑/↓ select · Enter open · 1-9,0 quick · Esc cancel</Text>
      </Box>
    )
  }

  if (stage === 'skill') {
    const { items, off } = visibleItems(skills, skillIdx)

    return (
      <Box flexDirection="column">
        <Text bold color={t.color.amber}>
          {selectedCat}
        </Text>

        <Text color={t.color.dim}>{skills.length} skill(s)</Text>
        {!skills.length ? <Text color={t.color.dim}>no skills in this category</Text> : null}
        {off > 0 && <Text color={t.color.dim}> ↑ {off} more</Text>}

        {items.map((row, i) => {
          const idx = off + i

          return (
            <Text color={skillIdx === idx ? t.color.cornsilk : t.color.dim} key={row}>
              {skillIdx === idx ? '▸ ' : '  '}
              {i + 1}. {row}
            </Text>
          )
        })}

        {off + VISIBLE < skills.length && <Text color={t.color.dim}> ↓ {skills.length - off - VISIBLE} more</Text>}
        <Text color={t.color.dim}>
          {skills.length ? '↑/↓ select · Enter open · 1-9,0 quick · Esc back' : 'Esc back'}
        </Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      <Text bold color={t.color.amber}>
        {info?.name ?? skillName}
      </Text>

      <Text color={t.color.dim}>{info?.category ?? selectedCat}</Text>
      {info?.description ? <Text color={t.color.cornsilk}>{info.description}</Text> : null}
      {info?.path ? <Text color={t.color.dim}>path: {info.path}</Text> : null}
      {!info && !err ? <Text color={t.color.dim}>loading…</Text> : null}
      {err ? <Text color={t.color.label}>error: {err}</Text> : null}
      {installing ? <Text color={t.color.amber}>installing…</Text> : null}

      <Text color={t.color.dim}>Enter install · i inspect · x install · Esc back</Text>
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
