import type { ScrollBoxHandle } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type RefObject, useEffect, useRef } from 'react'

import { $backtrackPrime, $pendingRewind } from '../app/backtrackStore.js'
import { $rewindFocus, $rewindScrollRestore, rewindFocusIndex } from '../app/rewindFocusStore.js'
import { useT } from '../i18n/useT.js'
import type { Msg } from '../types.js'

/** While backtrack browsing is open, keep the selected prompt in view: scroll so its row sits a few
 *  lines below the viewport top. The position browsing started from is remembered; Esc (back)
 *  returns to it, any other exit leaves the scroll where it is (sticky-bottom resumes on the next send). */
export function RewindFocusScroller({ messages, offsets, scrollRef }: RewindFocusScrollerProps) {
  const focus = useStore($rewindFocus)

  return (
    <FocusRowScroller
      active={Boolean(focus)}
      offsets={offsets}
      restore={takeRewindRestore}
      row={rewindFocusIndex(messages, focus)}
      scrollRef={scrollRef}
    />
  )
}

const takeRewindRestore = () => {
  const restore = $rewindScrollRestore.get()

  $rewindScrollRestore.set(false)

  return restore
}

/** Scroll so transcript `row` sits a few lines below the viewport top while `active`; the position
 *  from when it became active is remembered and returned to on close when `restore()` says so. */
export function FocusRowScroller({ active, offsets, restore, row, scrollRef }: FocusRowScrollerProps) {
  const top = row >= 0 ? offsets[row] : undefined
  const origin = useRef<null | { sticky: boolean; top: number }>(null)

  useEffect(() => {
    const scroll = scrollRef.current

    if (active && !origin.current && scroll) {
      origin.current = { sticky: scroll.isSticky(), top: scroll.getScrollTop() }
    }

    if (!active && origin.current) {
      const back = origin.current

      origin.current = null

      if (restore()) {
        back.sticky ? scroll?.scrollToBottom() : scroll?.scrollTo(back.top)
      }
    }
  }, [active, restore, scrollRef])

  useEffect(() => {
    if (top !== undefined) {
      scrollRef.current?.scrollTo(Math.max(0, top - 2))
    }
  }, [scrollRef, top])

  return null
}

/** The focused row's index in `messages`, -1 when not browsing (drawn in reverse video). */
export function useRewindFocusIndex(messages: readonly Msg[]) {
  return rewindFocusIndex(messages, useStore($rewindFocus))
}

/** The composer's one-line backtrack hint: the primed Esc (`esc again…`, only with an earlier
 *  prompt to go back to), else the armed edit, else null. */
export function useBacktrackHint(): null | string {
  const R = useT().prompt.rewind
  const prime = useStore($backtrackPrime)
  const pending = useStore($pendingRewind)

  if (prime?.count) {
    return R.prime
  }

  return pending ? R.pending(String(pending.target.turn)) : null
}

interface FocusRowScrollerProps {
  active: boolean
  offsets: ArrayLike<number>
  restore: () => boolean
  row: number
  scrollRef: RefObject<null | ScrollBoxHandle>
}

interface RewindFocusScrollerProps {
  messages: readonly Msg[]
  offsets: ArrayLike<number>
  scrollRef: RefObject<null | ScrollBoxHandle>
}
