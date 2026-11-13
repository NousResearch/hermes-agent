import { useState } from 'react'
import { Plus, X } from 'lucide-react'
import { Button } from '@nous-research/ui/ui/components/button'
import { Input } from '@nous-research/ui/ui/components/input'

interface Row {
  id: number
  value: string
}

// Row identity for React keys and focus.
let rowSeq = 0

/** Allowlists are stored as one comma-separated env value; a pasted list may also use newlines. */
export function splitAllowlist(raw: string | null | undefined): string[] {
  return (raw || '')
    .split(/[,\n]/)
    .map(entry => entry.trim())
    .filter(Boolean)
}

export function joinAllowlist(entries: string[]): string {
  return [...new Set(entries.map(entry => entry.trim()).filter(Boolean))].join(',')
}

function toRows(raw: string | null | undefined): Row[] {
  const entries = splitAllowlist(raw)
  return (entries.length ? entries : ['']).map(value => ({ id: rowSeq++, value }))
}

/** One input per allowed user/ID, with + to add and × to remove; reports one comma-separated value. */
export function AllowlistInput({
  id,
  label,
  value,
  onChange,
  invalid
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  invalid?: boolean
}) {
  const [rows, setRows] = useState<Row[]>(() => toRows(value))
  const [focusId, setFocusId] = useState<number | null>(null)

  const commit = (next: Row[]) => {
    setRows(next)
    onChange(joinAllowlist(next.map(row => row.value)))
  }

  const change = (rowId: number, text: string) => {
    // A pasted "123, 456" becomes one box per entry.
    const parts = /[,\n]/.test(text) ? splitAllowlist(text) : [text]
    const index = rows.findIndex(row => row.id === rowId)
    const replacement = parts.length
      ? parts.map((part, i) => ({ id: i === 0 ? rowId : rowSeq++, value: part }))
      : [{ id: rowId, value: '' }]
    commit([...rows.slice(0, index), ...replacement, ...rows.slice(index + 1)])
  }

  const add = () => {
    const rowId = rowSeq++
    setFocusId(rowId)
    setRows([...rows, { id: rowId, value: '' }])
  }

  const remove = (rowId: number) => {
    const next = rows.filter(row => row.id !== rowId)
    commit(next.length ? next : [{ id: rowSeq++, value: '' }])
  }

  const lone = rows.length === 1 && !rows[0].value

  return (
    <div className="grid gap-1.5" data-slot="allowlist-input">
      {rows.map((row, index) => (
        <div className="flex items-center gap-2" key={row.id}>
          <Input
            id={index === 0 ? id : undefined}
            aria-label={`${label} ${index + 1}`}
            aria-invalid={invalid}
            autoFocus={row.id === focusId}
            type="text"
            className="text-base leading-6 sm:text-xs sm:leading-4"
            placeholder="Enter an ID"
            value={row.value}
            onChange={e => change(row.id, e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && row.value.trim()) {
                e.preventDefault()
                add()
              }
            }}
          />
          <Button ghost size="icon" aria-label="Remove" title="Remove" disabled={lone} onClick={() => remove(row.id)}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      ))}
      <div>
        <Button ghost size="sm" onClick={add} prefix={<Plus className="h-3.5 w-3.5" />}>
          Add another
        </Button>
      </div>
    </div>
  )
}
