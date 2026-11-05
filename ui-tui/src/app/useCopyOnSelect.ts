import type { useSelection } from '@hermes/ink'
import { useEffect, useRef } from 'react'

type CopySelection = Pick<
  ReturnType<typeof useSelection>,
  'copySelectionNoClear' | 'getState' | 'hasSelection' | 'subscribe' | 'version'
>

/** Listen outside React so dragging does not rerender the app on every mouse move. */
export function useCopyOnSelect(selection: CopySelection, enabled: boolean) {
  const lastCopiedVersion = useRef(-1)

  useEffect(() => {
    if (!enabled) {
      return
    }

    return selection.subscribe(() => {
      if (!selection.hasSelection()) {
        return
      }

      const state = selection.getState() as { isDragging?: boolean } | null

      if (state?.isDragging) {
        return
      }

      const version = selection.version()

      if (version === lastCopiedVersion.current) {
        return
      }

      lastCopiedVersion.current = version
      void selection.copySelectionNoClear()
    })
  }, [enabled, selection])
}
