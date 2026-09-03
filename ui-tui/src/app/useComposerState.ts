import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { useStdin, withInkSuspended } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useRef, useState } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import { droppedTokens, imageToken, nextImageIndex } from '../domain/attachments.js'
import type { ClipboardPasteResponse, ImageAttachResponse, InputDetectDropResponse } from '../gatewayTypes.js'
import { useCompletion } from '../hooks/useCompletion.js'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { useQueue } from '../hooks/useQueue.js'
import { isUsableClipboardText, readClipboardText } from '../lib/clipboard.js'
import { resolveEditor } from '../lib/editor.js'
import { readOsc52Clipboard } from '../lib/osc52.js'
import { isRemoteShellSession } from '../lib/terminalSetup.js'
import { pasteTokenLabel, stripTrailingPasteNewlines } from '../lib/text.js'

import type {
  ComposerPasteResult,
  ComposerToken,
  MaybePromise,
  StateSetter,
  UseComposerStateOptions,
  UseComposerStateResult
} from './interfaces.js'
import { $isBlocked } from './overlayStore.js'
import { getUiState } from './uiStore.js'

const TOKEN_MAX_COUNT = 32
const TOKEN_MAX_TOTAL_BYTES = 4 * 1024 * 1024

const trimTokens = (tokens: ComposerToken[]): ComposerToken[] => {
  let total = 0
  const out: ComposerToken[] = []

  for (let i = tokens.length - 1; i >= 0; i--) {
    const token = tokens[i]!
    const size = token.text?.length ?? 0

    if (out.length >= TOKEN_MAX_COUNT || total + size > TOKEN_MAX_TOTAL_BYTES) {
      break
    }

    total += size
    out.unshift(token)
  }

  return out.length === tokens.length ? tokens : out
}

/** Insert text at the cursor position, adding spacing to separate from adjacent non-whitespace. */
function insertAtCursor(value: string, cursor: number, text: string): { cursor: number; value: string } {
  const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
  const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
  const insert = `${lead}${text}${tail}`

  return {
    cursor: cursor + insert.length,
    value: value.slice(0, cursor) + insert + value.slice(cursor)
  }
}

/**
 * Quick client-side heuristic to detect text that looks like a dropped file path.
 * When this returns true the composer sends RPC calls to the server for actual
 * validation. Keep in sync with _detect_file_drop() in cli.py — see that
 * function for the canonical prefix list.
 */
export function looksLikeDroppedPath(text: string): boolean {
  const trimmed = text.trim()

  if (!trimmed || trimmed.includes('\n')) {
    return false
  }

  // file:// URIs, relative, home-relative, quoted, and Windows drive paths
  if (
    trimmed.startsWith('file://') ||
    trimmed.startsWith('~/') ||
    trimmed.startsWith('./') ||
    trimmed.startsWith('../') ||
    trimmed.startsWith('"/') ||
    trimmed.startsWith("'/") ||
    trimmed.startsWith('"~') ||
    trimmed.startsWith("'~") ||
    /^[A-Za-z]:[/\\]/.test(trimmed) ||
    /^["'][A-Za-z]:[/\\]/.test(trimmed)
  ) {
    return true
  }

  // Bare absolute paths (start with /) — require a second '/' or a '.' to avoid
  // false positives on short strings like "/api" or "/help" which would trigger
  // unnecessary RPC round-trips.
  if (trimmed.startsWith('/')) {
    const rest = trimmed.slice(1)

    return rest.includes('/') || rest.includes('.')
  }

  return false
}

export function useComposerState({ gw, submitRef, sys }: UseComposerStateOptions): UseComposerStateResult {
  const [input, setInputState] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [tokens, setTokens] = useState<ComposerToken[]>([])
  // Tokens and the input line are read from keystroke handlers that run several
  // times before React re-renders, so the refs — not the state — are the source
  // of truth for "what is in the composer right now".
  const inputRef = useRef('')
  const tokensRef = useRef<ComposerToken[]>([])

  const setInput = useCallback<StateSetter<string>>(next => {
    inputRef.current = typeof next === 'function' ? next(inputRef.current) : next
    setInputState(inputRef.current)
  }, [])

  const setComposerTokens = useCallback<StateSetter<ComposerToken[]>>(next => {
    tokensRef.current = typeof next === 'function' ? next(tokensRef.current) : next
    setTokens(tokensRef.current)
  }, [])

  const isBlocked = useStore($isBlocked)
  const { querier } = useStdin() as { querier: Parameters<typeof readOsc52Clipboard>[0] }

  const {
    queueRef,
    queueEditRef,
    queuedDisplay,
    queueEditIdx,
    enqueue,
    dequeue,
    removeQ,
    replaceQ,
    setQueueEdit,
    syncQueue
  } = useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, isBlocked, gw)

  const clearIn = useCallback(() => {
    setInput('')
    setInputBuf([])
    setComposerTokens([])
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }, [historyDraftRef, setComposerTokens, setHistoryIdx, setInput, setQueueEdit])

  /**
   * Deleting an `[[ Image N ]]` token IS how you unattach the image — there is
   * no separate control. Reconcile on every edit so the gateway's
   * `attached_images` never outlives the token the user just erased, which is
   * what used to make a stale image ride along on the next unrelated turn.
   */
  const syncTokens = useCallback(
    (value: string) => {
      const gone = droppedTokens(tokensRef.current, value)

      if (!gone.length) {
        return
      }

      for (const token of gone) {
        if (token.kind === 'image') {
          void gw.request('image.detach', { path: token.path, session_id: getUiState().sid }).catch(() => {})
        }
      }

      setComposerTokens(prev => prev.filter(token => !gone.includes(token)))
    },
    [gw, setComposerTokens]
  )

  /**
   * Attach an image the gateway already resolved: a token at the cursor,
   * followed by whatever non-path text came along with it (a drag-drop paste
   * of `~/shot.png look at this` keeps the caption).
   */
  const attachImageToken = useCallback(
    (attached: ImageAttachResponse & { path?: string }, value: string, cursor: number): ComposerPasteResult => {
      const index = nextImageIndex(tokensRef.current)
      const label = imageToken(index)

      setComposerTokens(prev => trimTokens([...prev, { index, kind: 'image', label, path: attached.path ?? '' }]))

      const withToken = insertAtCursor(value, cursor, label)
      const remainder = attached.remainder?.trim() ?? ''

      return remainder ? insertAtCursor(withToken.value, withToken.cursor, remainder) : withToken
    },
    [setComposerTokens]
  )

  /**
   * Pull an image off the system clipboard into the composer as a token.
   *
   * `quiet` is the empty-bracketed-paste probe: the terminal delivers an image
   * paste as zero text, so we speculatively ask the gateway and stay silent if
   * there was nothing there. An explicit `/paste` reports the miss.
   */
  const pasteClipboardImage = useCallback(
    async (value: string, cursor: number, quiet: boolean): Promise<ComposerPasteResult | null> => {
      const sid = getUiState().sid

      if (!sid) {
        return null
      }

      const r = await gw
        .request<ClipboardPasteResponse & { path?: string }>('clipboard.paste', { session_id: sid })
        .catch(() => null)

      if (r?.attached) {
        return attachImageToken(r, value, cursor)
      }

      if (!quiet) {
        sys(r?.message || 'No image found in clipboard')
      }

      return null
    },
    [attachImageToken, gw, sys]
  )

  const handleResolvedPaste = useCallback(
    async ({ bracketed, cursor, text, value }: Omit<PasteEvent, 'hotkey'>): Promise<ComposerPasteResult | null> => {
      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText || !/[^\n]/.test(cleanedText)) {
        return bracketed ? pasteClipboardImage(value, cursor, true) : null
      }

      const sid = getUiState().sid

      if (sid && looksLikeDroppedPath(cleanedText)) {
        try {
          const attached = await gw.request<ImageAttachResponse>('image.attach', {
            path: cleanedText,
            session_id: sid
          })

          if (attached?.name) {
            // Drop an `[[ Image N ]]` token where the path was typed. The old
            // path printed a notice above the status bar and left the composer
            // untouched, so the only trace of the attachment lived outside the
            // input the user was editing.
            return attachImageToken(attached, value, cursor)
          }
        } catch {
          // Fall back to generic file-drop detection below.
        }

        try {
          const dropped = await gw.request<InputDetectDropResponse>('input.detect_drop', {
            session_id: sid,
            text: cleanedText
          })

          if (dropped?.matched && dropped.text) {
            return insertAtCursor(value, cursor, dropped.text)
          }
        } catch {
          // Fall through to normal text paste behavior.
        }
      }

      const lineCount = cleanedText.split('\n').length
      const pasteCollapseLines = getUiState().pasteCollapseLines
      const pasteCollapseChars = getUiState().pasteCollapseChars
      const linesHit = pasteCollapseLines > 0 && lineCount >= pasteCollapseLines
      const charsHit = pasteCollapseChars > 0 && cleanedText.length >= pasteCollapseChars

      if (!linesHit && !charsHit) {
        return {
          cursor: cursor + cleanedText.length,
          value: value.slice(0, cursor) + cleanedText + value.slice(cursor)
        }
      }

      const label = pasteTokenLabel(cleanedText, lineCount)
      const inserted = insertAtCursor(value, cursor, label)

      setComposerTokens(prev => trimTokens([...prev, { kind: 'paste', label, text: cleanedText }]))

      void gw
        .request<{ path?: string }>('paste.collapse', { text: cleanedText })
        .then(r => {
          const path = r?.path

          if (!path) {
            return
          }

          setComposerTokens(prev => prev.map(t => (t.label === label ? { ...t, path } : t)))
        })
        .catch(() => {})

      return inserted
    },
    [attachImageToken, gw, pasteClipboardImage, setComposerTokens]
  )

  const handleTextPaste = useCallback(
    ({
      bracketed,
      cursor,
      hotkey,
      text,
      value
    }: PasteEvent): MaybePromise<ComposerPasteResult | null> => {
      if (hotkey) {
        const preferOsc52 = isRemoteShellSession(process.env)

        const readPreferredText = preferOsc52
          ? readOsc52Clipboard(querier).then(async osc52Text => {
              if (isUsableClipboardText(osc52Text)) {
                return osc52Text
              }

              return readClipboardText()
            })
          : readClipboardText().then(async clipText => {
              if (isUsableClipboardText(clipText)) {
                return clipText
              }

              return readOsc52Clipboard(querier)
            })

        return readPreferredText.then(async preferredText => {
          if (isUsableClipboardText(preferredText)) {
            return handleResolvedPaste({ bracketed: false, cursor, text: preferredText, value })
          }

          // No text on the clipboard — an image paste looks exactly like this.
          return pasteClipboardImage(value, cursor, false)
        })
      }

      return handleResolvedPaste({ bracketed: !!bracketed, cursor, text, value })
    },
    [handleResolvedPaste, pasteClipboardImage, querier]
  )

  /**
   * `/paste` and `/image` attach without a cursor of their own — the token
   * lands at the end of whatever is currently typed.
   */
  const appendAttachment = useCallback(
    (attach: (value: string, cursor: number) => Promise<ComposerPasteResult | null>) => {
      const current = inputRef.current

      void attach(current, current.length).then(next => {
        if (next) {
          setInput(next.value)
        }
      })
    },
    [setInput]
  )

  const attachClipboardImage = useCallback(
    () => appendAttachment((value, cursor) => pasteClipboardImage(value, cursor, false)),
    [appendAttachment, pasteClipboardImage]
  )

  const attachImagePath = useCallback(
    (path: string) =>
      appendAttachment(async (value, cursor) => {
        const sid = getUiState().sid

        if (!sid || !path.trim()) {
          return null
        }

        const attached = await gw
          .request<ImageAttachResponse & { path?: string }>('image.attach', { path, session_id: sid })
          .catch((e: Error) => {
            sys(`error: ${e.message}`)

            return null
          })

        return attached?.name ? attachImageToken(attached, value, cursor) : null
      }),
    [appendAttachment, attachImageToken, gw, sys]
  )

  const openEditor = useCallback(async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hermes-'))
    const file = join(dir, 'prompt.md')
    const [cmd, ...args] = resolveEditor()

    writeFileSync(file, [...inputBuf, input].join('\n'))

    let exitCode: null | number = null

    await withInkSuspended(async () => {
      exitCode = spawnSync(cmd!, [...args, file], { stdio: 'inherit' }).status
    })

    try {
      if (exitCode !== 0) {
        return
      }

      const text = readFileSync(file, 'utf8').trimEnd()

      if (!text) {
        return
      }

      setInput('')
      setInputBuf([])
      submitRef.current(text)
    } finally {
      rmSync(dir, { force: true, recursive: true })
    }
  }, [input, inputBuf, setInput, submitRef])

  const actions = useMemo(
    () => ({
      attachClipboardImage,
      attachImagePath,
      clearIn,
      dequeue,
      enqueue,
      handleTextPaste,
      openEditor,
      pushHistory,
      removeQueue: removeQ,
      replaceQueue: replaceQ,
      setCompIdx,
      setComposerTokens,
      setHistoryIdx,
      setInput,
      setInputBuf,
      setQueueEdit,
      syncQueue,
      syncTokens
    }),
    [
      attachClipboardImage,
      attachImagePath,
      clearIn,
      dequeue,
      enqueue,
      handleTextPaste,
      openEditor,
      pushHistory,
      removeQ,
      replaceQ,
      setCompIdx,
      setComposerTokens,
      setHistoryIdx,
      setInput,
      setQueueEdit,
      syncQueue,
      syncTokens
    ]
  )

  const refs = useMemo(
    () => ({
      historyDraftRef,
      historyRef,
      queueEditRef,
      queueRef,
      submitRef,
      tokensRef
    }),
    [historyDraftRef, historyRef, queueEditRef, queueRef, submitRef]
  )

  const state = useMemo(
    () => ({
      compIdx,
      compReplace,
      completions,
      historyIdx,
      input,
      inputBuf,
      queueEditIdx,
      queuedDisplay,
      tokens
    }),
    [compIdx, compReplace, completions, historyIdx, input, inputBuf, queueEditIdx, queuedDisplay, tokens]
  )

  return {
    actions,
    refs,
    state
  }
}
