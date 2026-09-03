import { useState } from 'react'

import { Markdown } from '@/components/Markdown'
import type { WisdomDraftReview } from '@/lib/api'
import { Button } from '@nous-research/ui/ui/components/button'
import { WisdomManifestEditor } from './WisdomManifestEditor'

interface Props {
  file: WisdomDraftReview['files'][number]
  value: string
  disabled?: boolean
  onChange: (value: string) => void
}

export function WisdomFileEditor({ file, value, disabled = false, onChange }: Props) {
  const [mode, setMode] = useState<'source' | 'preview'>('source')

  if (file.path === 'skill.manifest.json') {
    return (
      <details className="border-t border-border py-3" open>
        <summary className="cursor-pointer font-mono text-xs">
          {file.path} · server-reviewed {file.hash}
        </summary>
        <WisdomManifestEditor value={value} disabled={disabled} onChange={onChange} />
      </details>
    )
  }

  const supportsPreview = file.path.toLowerCase().endsWith('.md')

  return (
    <details className="border-t border-border py-3" open>
      <summary className="cursor-pointer font-mono text-xs">
        {file.path} · server-reviewed {file.hash}
      </summary>
      {supportsPreview && (
        <div className="mt-3 flex gap-2" aria-label={`${file.path} editor mode`}>
          <Button size="xs" outlined={mode !== 'source'} onClick={() => setMode('source')}>
            Source
          </Button>
          <Button size="xs" outlined={mode !== 'preview'} onClick={() => setMode('preview')}>
            Preview
          </Button>
        </div>
      )}
      {supportsPreview && mode === 'preview' ? (
        <div className="mt-3 min-h-48 max-h-[32rem] overflow-auto border border-border bg-muted/10 p-4">
          <Markdown content={value} />
        </div>
      ) : (
        <textarea
          aria-label={`Edit ${file.path}`}
          className="mt-3 min-h-64 max-h-[32rem] w-full resize-y border border-border bg-background/40 px-3 py-2 font-mono text-xs leading-relaxed shadow-sm placeholder:text-muted-foreground focus-visible:border-foreground/25 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 disabled:cursor-not-allowed disabled:opacity-60"
          disabled={disabled}
          spellCheck={false}
          value={value}
          onChange={event => onChange(event.target.value)}
        />
      )}
    </details>
  )
}
