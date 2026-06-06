import { useEffect, useState } from 'react'
import { Box, Text } from '@hermes/ink'

import type { Theme } from '../theme.js'

export interface AttachedFileMeta {
  id: string
  name: string
  stored_path: string
  mime_type: string
  size_bytes: number
  kind: 'IMAGE' | 'PDF' | 'TEXT' | 'BINARY'
}

export interface AttachmentsBarProps {
  // The gateway client to call file.list. Optional — if not provided
  // the bar is purely presentational (e.g. in tests or previews).
  fetch?: (sessionId: string) => Promise<AttachedFileMeta[]>
  sessionId: string | null
  t: Theme
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function iconFor(mime: string): string {
  if (mime.startsWith('image/')) return '🖼'
  if (mime === 'application/pdf') return '📕'
  if (mime.startsWith('text/') || mime.includes('json') || mime.includes('yaml') || mime.includes('toml')) {
    return '📄'
  }
  return '📎'
}

export function AttachmentsBar({ fetch, sessionId, t }: AttachmentsBarProps) {
  const [files, setFiles] = useState<AttachedFileMeta[]>([])

  useEffect(() => {
    if (!sessionId || !fetch) {
      setFiles([])
      return
    }
    let cancelled = false
    void fetch(sessionId)
      .then(list => {
        if (!cancelled) {
          setFiles(list)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFiles([])
        }
      })
    return () => {
      cancelled = true
    }
  }, [fetch, sessionId])

  if (!sessionId || files.length === 0) {
    return null
  }

  const shown = files.slice(0, 6)
  const overflow = files.length - shown.length

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color={t.color.muted} dimColor>{`📎 Attached (${files.length})`}</Text>
      {shown.map(f => (
        <Box key={f.id} flexDirection="row" justifyContent="space-between">
          <Text>
            {`  ${iconFor(f.mime_type)} ${f.name}`}
          </Text>
          <Text color={t.color.muted} dimColor>
            {`${formatSize(f.size_bytes)} · ${f.kind.toLowerCase()}`}
          </Text>
        </Box>
      ))}
      {overflow > 0 ? (
        <Text color={t.color.muted} dimColor>{`  …and ${overflow} more`}</Text>
      ) : null}
    </Box>
  )
}
