import { useCallback, useMemo, useRef, useState } from 'react'

import { placeInputCursor } from '../app/inputSelectionStore.js'
import type { ComposerToken } from '../app/interfaces.js'
import {
  formatStashAge,
  PromptStash,
  resolveCtrlS,
  type StashAction,
  type StashPayload
} from '../domain/promptStash.js'
import { messages } from '../i18n/runtime.js'

export interface StashPanelRow {
  age: string
  preview: string
}

export interface PromptStashState {
  /** Placeholder hint the classic CLI shows while a draft is parked ('' when none). */
  hint: string
  /** Status-bar indicator, `📌 N` / `📌 N ▲` / ''. */
  indicator: string
  panelCursor: number
  panelOpen: boolean
  /** Browse-panel rows, newest first. */
  rows: StashPanelRow[]
}

export interface PromptStashActions {
  /** Ctrl+S with the panel open closes it (the classic CLI's escape hatch). */
  closePanel: () => void
  deleteAtCursor: () => boolean
  moveCursor: (delta: number) => void
  restoreAtCursor: () => boolean
  /** One Ctrl+S press. Returns the action taken so the caller can react to it. */
  toggle: () => StashAction
}

interface UsePromptStashOptions {
  /** Clear the composer the way `buf.reset()` does. */
  clear: () => void
  getInput: () => string
  getTokens: () => ComposerToken[]
  setComposerTokens: (next: ComposerToken[] | ((prev: ComposerToken[]) => ComposerToken[])) => void
  setHistoryIdx: (next: null | number) => void
  setInput: (next: string | ((prev: string) => string)) => void
  setQueueEdit: (next: null | number) => void
}

/**
 * Session-scoped Ctrl+S stash for the Ink composer.
 *
 * The classic CLI keeps the state machine in `hermes_cli/prompt_stash.py` and only owns the
 * keybinding and the drawing. This hook is the TUI's half of that split: the decision table
 * lives in `domain/promptStash.ts` (a port of the Python one), and everything here applies
 * its outcome to composer state.
 *
 * The stash itself is a ref: it is mutable session state read from a keystroke handler that
 * runs several times before React re-renders, exactly like `queueRef`. A version counter is
 * what makes the panel, the indicator and the placeholder re-render.
 */
export function usePromptStash({
  clear,
  getInput,
  getTokens,
  setComposerTokens,
  setHistoryIdx,
  setInput,
  setQueueEdit
}: UsePromptStashOptions): { actions: PromptStashActions; state: PromptStashState } {
  const stash = useRef(new PromptStash()).current
  const [, bumpVersion] = useState(0)
  const repaint = useCallback(() => bumpVersion(v => v + 1), [])

  const applyPayload = useCallback(
    (payload: null | StashPayload): boolean => {
      if (!payload) {
        return false
      }

      // Attachments are extended, never replaced: something attached since the stash was
      // taken must survive. The caret parks at the end of the draft, as the classic CLI's
      // `buf.cursor_position = len(text)` does.
      if (payload.tokens.length) {
        setComposerTokens(prev => [...prev, ...payload.tokens])
      }

      setHistoryIdx(null)
      setQueueEdit(null)
      placeInputCursor(payload.text, payload.text.length)
      setInput(payload.text)

      return true
    },
    [setComposerTokens, setHistoryIdx, setInput, setQueueEdit]
  )

  const toggle = useCallback((): StashAction => {
    const { action, payload } = resolveCtrlS(stash, getInput(), getTokens())

    if (action === 'stashed') {
      // `buf.reset()` parity: draft, chips and any in-flight edit state all go.
      clear()
    } else if (action === 'restored') {
      applyPayload(payload)
    }

    repaint()

    return action
  }, [applyPayload, clear, getInput, getTokens, repaint, stash])

  const actions = useMemo<PromptStashActions>(() => {
    // Every panel action mutates the ref'd stash, then repaints the badge / panel / hint.
    const thenRepaint =
      <A extends unknown[], R>(fn: (...args: A) => R) =>
      (...args: A): R => {
        const result = fn(...args)

        repaint()

        return result
      }

    return {
      closePanel: thenRepaint(() => stash.closePanel()),
      deleteAtCursor: thenRepaint(() => stash.deleteAtCursor()),
      moveCursor: thenRepaint((delta: number) => void stash.moveCursor(delta)),
      restoreAtCursor: thenRepaint(() => applyPayload(stash.restoreAtCursor())),
      toggle
    }
  }, [applyPayload, repaint, stash, toggle])

  const stashStrings = messages().composer.stash
  const now = Date.now()

  const state: PromptStashState = {
    hint: stash.placeholderHint({ browse: stashStrings.browseHint, restore: stashStrings.restoreHint }),
    indicator: stash.indicator(),
    panelCursor: stash.panelCursor,
    panelOpen: stash.panelOpen,
    rows: stash.panelRows().map(entry => ({
      age: formatStashAge(entry.stashedAt, now, {
        hours: stashStrings.ageHours,
        justNow: stashStrings.ageJustNow,
        minutes: stashStrings.ageMinutes,
        seconds: stashStrings.ageSeconds
      }),
      preview: entry.preview
    }))
  }

  return { actions, state }
}
