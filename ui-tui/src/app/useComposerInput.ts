import { useCallback } from 'react'

import type { ComposerActions, ComposerRefs, ComposerState, StateSetter } from './interfaces.js'
import { shouldDetachEditedHistoryInput } from './useInputHandlers.js'

/** Reconcile user edits, never programmatic draft/history replacement. */
export function useComposerInput(actions: ComposerActions, refs: ComposerRefs, historyIdx: ComposerState['historyIdx']) {
  return useCallback<StateSetter<string>>(next => {
    actions.editInput(prev => {
      const value = typeof next === 'function' ? next(prev) : next
      actions.syncTokens(value)

      if (shouldDetachEditedHistoryInput(historyIdx, refs.historyRef.current, value)) {
        refs.historyDraftRef.current = value
        actions.setHistoryIdx(null)
      }

      return value
    })
  }, [actions, refs, historyIdx])
}
