'use client'

import {
  Children,
  isValidElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode
} from 'react'
import { createPortal } from 'react-dom'

import { cn } from '@/lib/utils'

const TRIGGER_CN =
  'flex h-9 w-full items-center justify-between gap-2 ' +
  'border border-midground/15 bg-background/40 px-3 py-1 ' +
  'font-courier text-sm text-left text-midground transition-colors ' +
  'hover:border-midground/25 ' +
  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/30 focus-visible:border-midground/30 ' +
  'disabled:cursor-not-allowed disabled:opacity-50 ' +
  'cursor-pointer'

// Rendered through a portal onto document.body (see below), so position is
// applied inline. `position: fixed` here only as a fallback for SSR.
const LISTBOX_CN =
  'z-50 max-h-60 overflow-auto ' +
  'border border-midground/15 bg-background-base text-midground shadow-lg'

/**
 * Dashboard select styled with the shared design tokens.
 *
 * This is the dashboard's own copy of `@nous-research/ui`'s `Select`, wired in
 * via Vite's resolve.alias (see `web/vite.config.ts`) so the dashboard no
 * longer depends on the published component.
 *
 * Why a local copy: the upstream `@nous-research/ui` `Select` renders its
 * option list as an `absolute` node *inside* the trigger's `relative`
 * container. Whenever that container sits in a scrolling or `overflow`
 * wrapper (the composer bar, sidebars, dialogs), a new stacking context is
 * created and the menu gets clipped or pushed underneath — so expanding a
 * multi-select shows nothing. The desktop app already avoids this by portaling
 * (Radix `Portal`). This component does the same: the option list is rendered
 * through `createPortal(document.body)` and positioned with `position: fixed`
 * under the trigger, so it always paints above its ancestors.
 */
export function Select({
  children,
  className,
  disabled,
  id,
  onValueChange,
  placeholder,
  style,
  value
}: SelectProps) {
  const [open, setOpen] = useState(false)
  const [highlightedIndex, setHighlightedIndex] = useState(-1)
  const [menuRect, setMenuRect] = useState<{
    top: number
    left: number
    width: number
  } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const options = useMemo(() => collectOptions(children), [children])
  const selected = options.find(o => o.value === value)
  const displayLabel = selected?.label ?? placeholder ?? value ?? ''

  const close = useCallback(() => {
    setOpen(false)
    setHighlightedIndex(-1)
  }, [])

  useEffect(() => {
    if (!open) return
    // Measure the trigger every time the menu opens so the portaled listbox
    // docks exactly under it, even if an ancestor scrolled/overflow container
    // would otherwise clip it (the menu is not a DOM descendant of the trigger).
    const r = triggerRef.current?.getBoundingClientRect()
    if (r) setMenuRect({ top: r.bottom + 4, left: r.left, width: r.width })
  }, [open])

  useEffect(() => {
    if (!open) return
    const ac = new AbortController()
    document.addEventListener(
      'mousedown',
      e => {
        const t = e.target as Node
        // The listbox is portaled to body, so count it as "inside" too —
        // clicking an option must not close the menu before its onClick runs.
        const inside =
          containerRef.current?.contains(t) || listRef.current?.contains(t)
        if (!inside) close()
      },
      { signal: ac.signal }
    )
    return () => ac.abort()
  }, [open, close])

  useEffect(() => {
    if (!open || highlightedIndex < 0) return
    const el = listRef.current?.children[highlightedIndex] as
      | HTMLElement
      | undefined
    el?.scrollIntoView({ block: 'nearest' })
  }, [open, highlightedIndex])

  const handleKeyDown = (e: KeyboardEvent) => {
    if (disabled) return
    switch (e.key) {
      case 'Enter':
      case ' ':
        e.preventDefault()
        if (!open) {
          setOpen(true)
          setHighlightedIndex(options.findIndex(o => o.value === value))
        } else if (highlightedIndex >= 0 && options[highlightedIndex]) {
          onValueChange?.(options[highlightedIndex].value)
          close()
        }
        break
      case 'ArrowDown':
        e.preventDefault()
        if (!open) {
          setOpen(true)
          setHighlightedIndex(options.findIndex(o => o.value === value))
        } else {
          setHighlightedIndex(i => Math.min(i + 1, options.length - 1))
        }
        break
      case 'ArrowUp':
        e.preventDefault()
        if (open) setHighlightedIndex(i => Math.max(i - 1, 0))
        break
      case 'Home':
        if (open) {
          e.preventDefault()
          setHighlightedIndex(0)
        }
        break
      case 'End':
        if (open) {
          e.preventDefault()
          setHighlightedIndex(options.length - 1)
        }
        break
      case 'Escape':
        e.preventDefault()
        close()
        break
    }
  }

  return (
    <div
      className={cn('relative', className)}
      id={id}
      ref={containerRef}
      style={style}
    >
      <button
        aria-expanded={open}
        aria-haspopup="listbox"
        className={TRIGGER_CN}
        disabled={disabled}
        onClick={() => !disabled && setOpen(o => !o)}
        onKeyDown={handleKeyDown}
        ref={triggerRef}
        role="combobox"
        type="button"
      >
        <span className={cn('truncate', !selected && 'text-midground/50')}>
          {displayLabel}
        </span>

        <ChevronDownGlyph
          className={cn(
            'size-3 shrink-0 text-midground/60 transition-transform',
            open && 'rotate-180'
          )}
        />
      </button>

      {open &&
        menuRect &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            className={LISTBOX_CN}
            ref={listRef}
            role="listbox"
            style={{
              position: 'fixed',
              top: menuRect.top,
              left: menuRect.left,
              width: menuRect.width
            }}
          >
            {options.map((opt, i) => {
              const isSelected = opt.value === value
              const isHighlighted = i === highlightedIndex

              return (
                <div
                  aria-selected={isSelected}
                  className={cn(
                    'flex cursor-pointer items-center gap-2 px-3 py-2',
                    'font-courier text-sm transition-colors',
                    isHighlighted && 'bg-midground/10',
                    isSelected ? 'text-midground' : 'text-midground/70'
                  )}
                  key={opt.value}
                  onClick={() => {
                    onValueChange?.(opt.value)
                    close()
                  }}
                  onMouseEnter={() => setHighlightedIndex(i)}
                  role="option"
                >
                  <CheckGlyph
                    className={cn(
                      'size-3 shrink-0',
                      isSelected ? 'opacity-100' : 'opacity-0'
                    )}
                  />
                  <span className="truncate">{opt.label}</span>
                </div>
              )
            })}
          </div>,
          document.body
        )}
    </div>
  )
}

// Marker component — `Select` reads `value`/`children` from its tree.
// Renders nothing on its own.
export function SelectOption(_props: SelectOptionProps) {
  return null
}

const ChevronDownGlyph = ({ className }: { className?: string }) => (
  <svg
    aria-hidden
    className={className}
    fill="none"
    stroke="currentColor"
    strokeLinecap="square"
    strokeWidth={1.5}
    viewBox="0 0 12 12"
  >
    <path d="M2.5 4.5 6 8l3.5-3.5" />
  </svg>
)

const CheckGlyph = ({ className }: { className?: string }) => (
  <svg
    aria-hidden
    className={className}
    fill="none"
    stroke="currentColor"
    strokeLinecap="square"
    strokeWidth={1.5}
    viewBox="0 0 12 12"
  >
    <path d="m2.5 6.5 2.5 2.5L9.5 3.5" />
  </svg>
)

function collectOptions(children: ReactNode): SelectOptionData[] {
  const out: SelectOptionData[] = []
  Children.forEach(children, child => {
    if (!isValidElement(child)) return
    const el = child as ReactElement<{
      children?: ReactNode
      value?: unknown
    }>
    if (el.props.value !== undefined) {
      out.push({
        label:
          typeof el.props.children === 'string'
            ? el.props.children
            : String(el.props.value),
        value: String(el.props.value)
      })
    } else if (el.props.children) {
      out.push(...collectOptions(el.props.children))
    }
  })
  return out
}

interface SelectOptionData {
  label: string
  value: string
}

interface SelectOptionProps {
  children: ReactNode
  value: string
}

interface SelectProps {
  children?: ReactNode
  className?: string
  disabled?: boolean
  id?: string
  onValueChange?: (value: string) => void
  placeholder?: string
  style?: CSSProperties
  value?: string
}