import type { ScrollBoxHandle } from '@hermes/ink'
import { evictInkCaches, forceRedraw } from '@hermes/ink'
import type { RefObject } from 'react'

export const refreshSessionView = (stdout: NodeJS.WriteStream = process.stdout) => {
  evictInkCaches('all')
  forceRedraw(stdout)
}

export const scheduleResumeScrollToBottom = (
  scrollRef: RefObject<null | ScrollBoxHandle>,
  delays: readonly number[] = [0, 80, 240]
) => {
  const startedAt = Date.now()

  const timers = delays.map((delay, index) =>
    setTimeout(() => {
      const scroll = scrollRef.current

      if (!scroll) {
        return
      }

      const manuallyScrolledAfterResume = scroll.getLastManualScrollAt() > startedAt

      if (!manuallyScrolledAfterResume && (index === 0 || scroll.isSticky())) {
        scroll.scrollToBottom()

        if (index === 0) {
          refreshSessionView()
        }
      }
    }, delay)
  )

  return () => {
    for (const timer of timers) {
      clearTimeout(timer)
    }
  }
}
