import type { ScrollBoxHandle } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type RefObject, useEffect } from 'react'

import { $rewindFocus, rewindFocusIndex } from '../app/rewindFocusStore.js'
import type { Msg } from '../types.js'

/** While the rewind picker is open, keep the selected prompt in view: scroll so its row sits a few
 *  lines below the viewport top. Closing the picker leaves the scroll where it is (sticky-bottom
 *  resumes on the next send). */
export function RewindFocusScroller({ messages, offsets, scrollRef }: RewindFocusScrollerProps) {
  const focus = useStore($rewindFocus)
  const index = rewindFocusIndex(messages, focus)
  const top = index >= 0 ? offsets[index] : undefined

  useEffect(() => {
    if (top !== undefined) {
      scrollRef.current?.scrollTo(Math.max(0, top - 2))
    }
  }, [scrollRef, top])

  return null
}

/** The focused row's index in `messages`, -1 when the picker is closed (for the transcript's marker). */
export function useRewindFocusIndex(messages: readonly Msg[]) {
  return rewindFocusIndex(messages, useStore($rewindFocus))
}

interface RewindFocusScrollerProps {
  messages: readonly Msg[]
  offsets: ArrayLike<number>
  scrollRef: RefObject<null | ScrollBoxHandle>
}
