import { Box, Text } from 'ink'
import type { ReactNode } from 'react'

import type { Theme } from '../theme.js'

function MdInline({ t, text }: { t: Theme; text: string }) {
  const parts: ReactNode[] = []
  const re = /(\[(.+?)\]\((https?:\/\/[^\s)]+)\)|\*\*(.+?)\*\*|`([^`]+)`|\*(.+?)\*|(https?:\/\/[^\s]+))/g

  let last = 0
  let match: RegExpExecArray | null

  while ((match = re.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(
        <Text color={t.color.cornsilk} key={parts.length}>
          {text.slice(last, match.index)}
        </Text>
      )
    }

    if (match[2] && match[3]) {
      parts.push(
        <Text color={t.color.amber} key={parts.length} underline>
          {match[2]}
        </Text>
      )
    } else if (match[4]) {
      parts.push(
        <Text bold color={t.color.cornsilk} key={parts.length}>
          {match[4]}
        </Text>
      )
    } else if (match[5]) {
      parts.push(
        <Text color={t.color.amber} dimColor key={parts.length}>
          {match[5]}
        </Text>
      )
    } else if (match[6]) {
      parts.push(
        <Text color={t.color.cornsilk} italic key={parts.length}>
          {match[6]}
        </Text>
      )
    } else if (match[7]) {
      parts.push(
        <Text color={t.color.amber} key={parts.length} underline>
          {match[7]}
        </Text>
      )
    }

    last = match.index + match[0].length
  }

  if (last < text.length) {
    parts.push(
      <Text color={t.color.cornsilk} key={parts.length}>
        {text.slice(last)}
      </Text>
    )
  }

  return <Text>{parts.length ? parts : <Text color={t.color.cornsilk}>{text}</Text>}</Text>
}

export function Md({ compact, t, text }: { compact?: boolean; t: Theme; text: string }) {
  const lines = text.split('\n')
  const nodes: ReactNode[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]!
    const key = nodes.length

    if (compact && !line.trim()) {
      i++

      continue
    }

    if (line.startsWith('```')) {
      const lang = line.slice(3).trim()
      const block: string[] = []

      for (i++; i < lines.length && !lines[i]!.startsWith('```'); i++) {
        block.push(lines[i]!)
      }

      i++
      nodes.push(
        <Box flexDirection="column" key={key} paddingLeft={2}>
          {lang && <Text color={t.color.dim}>{'─ ' + lang}</Text>}
          {block.map((l, j) => (
            <Text color={t.color.cornsilk} key={j}>
              {l}
            </Text>
          ))}
        </Box>
      )

      continue
    }

    const heading = line.match(/^#{1,3}\s+(.*)/)

    if (heading) {
      nodes.push(
        <Text bold color={t.color.amber} key={key}>
          {heading[1]}
        </Text>
      )
      i++

      continue
    }

    const bullet = line.match(/^\s*[-*]\s(.*)/)

    if (bullet) {
      nodes.push(
        <Text key={key}>
          <Text color={t.color.dim}> • </Text>
          <MdInline t={t} text={bullet[1]!} />
        </Text>
      )
      i++

      continue
    }

    const numbered = line.match(/^\s*(\d+)\.\s(.*)/)

    if (numbered) {
      nodes.push(
        <Text key={key}>
          <Text color={t.color.dim}> {numbered[1]}. </Text>
          <MdInline t={t} text={numbered[2]!} />
        </Text>
      )
      i++

      continue
    }

    nodes.push(<MdInline key={key} t={t} text={line} />)
    i++
  }

  return <Box flexDirection="column">{nodes}</Box>
}
