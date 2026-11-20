import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { useStdin, withInkSuspended } from '@hermes/ink'
import type {
  ClipboardPasteResponse,
  ImageAttachResponse,
  InputDetectDropResponse
} from '@hermes/shared/gateway-types'
import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useRef, useState } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import { droppedTokens, imageToken, nextImageIndex } from '../domain/attachments.js'
import { collapsePasteSpan, expandPasteChip, insideSpan, type PasteSpan, shiftSpan } from '../domain/pasteExpand.js'
import { useCompletion } from '../hooks/useCompletion.js'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { useQueue } from '../hooks/useQueue.js'
import { t } from '../i18n/runtime.js'
import { isUsableClipboardText, readClipboardText } from '../lib/clipboard.js'
import { resolveEditor } from '../lib/editor.js'
import { stageClipboardImage, stageImagePath } from '../lib/imageAttachments.js'
import { readOsc52Clipboard } from '../lib/osc52.js'
import { isRemoteShellSession } from '../lib/terminalSetup.js'
import { pasteTokenLabel, stripTrailingPasteNewlines } from '../lib/text.js'

import { placeInputCursor } from './inputSelectionStore.js'
import type {
  ComposerPasteResult,
  ComposerToken,
  ComposerView,
  MaybePromise,
  StateSetter,
  UseComposerStateOptions,
  UseComposerStateResult
} from './interfaces.js'
import { $isBlocked } from './overlayStore.js'
import { captureDestination, isCurrentDestination, type SubmissionDestination } from './submissionDestination.js'
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

/** Whether a paste crosses the configured line or character collapse threshold. */
function shouldCollapsePaste(cleanedText: string, lineCount: number): boolean {
  const pasteCollapseLines = getUiState().pasteCollapseLines
  const pasteCollapseChars = getUiState().pasteCollapseChars
  const linesHit = pasteCollapseLines > 0 && lineCount >= pasteCollapseLines
  const charsHit = pasteCollapseChars > 0 && cleanedText.length >= pasteCollapseChars

  return linesHit || charsHit
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

/** The draft an in-flight paste or attach was started on. */
interface DraftMark {
  revision: number
  /** How many of this revision's landings the result's base value already holds. */
  seen: number
}

export function useComposerState({ gw, submitRef, sys }: UseComposerStateOptions): UseComposerStateResult {
  const [input, setInputState] = useState('')
  const [inputBuf, setInputBufState] = useState<string[]>([])
  // Bumped by every edit except a paste or attach result landing: a sibling
  // paste's own insert must not read as the user editing the draft away.
  const composerRevision = useRef(0)
  // Bumped whenever the draft is not just added to at the end (cleared, submitted, replaced, edited
  // mid-text): an edit's image restore follows the user's typing only within the draft it began on.
  const draftLife = useRef(0)
  // Bumped only when the draft ends (clearIn: cleared, submitted, Esc Esc): text arriving late from
  // browse (a paste still resolving, keys queued behind it) belongs to the draft it was typed into.
  // An edit inside that draft (backspace to empty, a `\`+Enter continuation, mid-text) keeps it.
  const draftEpoch = useRef(0)

  const inputBufRef = useRef<string[]>([])

  // Inserts results made on the current revision, in landing order, so a
  // later result moves its insertion point past the ones that landed first.
  const landings = useRef<Array<{ at: number; len: number; value: string }>>([])
  // The value a result's landing is about to write, so setInput knows it as its own.
  const landingValue = useRef<null | string>(null)
  // Tokens added by a result that has not landed in the text yet.
  const unlanded = useRef(new Set<ComposerToken>())

  const bumpRevision = useCallback(() => {
    composerRevision.current++
    landings.current = []
  }, [])

  // `base` is the value the result computes its cursor against. The input
  // line can hand over a value that has not caught up with landings written
  // straight to the composer yet, so match it rather than count every landing.
  const captureDraft = useCallback((base: string): DraftMark => {
    const seen =
      base === inputRef.current
        ? landings.current.length
        : landings.current.findLastIndex(landing => landing.value === base) + 1

    return { revision: composerRevision.current, seen }
  }, [])

  const isFresh = useCallback((draft: DraftMark) => draft.revision === composerRevision.current, [])

  const setInputBuf = useCallback<StateSetter<string[]>>(next => {
    bumpRevision()
    inputBufRef.current = typeof next === 'function' ? next(inputBufRef.current) : next
    setInputBufState(inputBufRef.current)
  }, [bumpRevision])

  const [tokens, setTokens] = useState<ComposerToken[]>([])
  // Tokens and the input line are read from keystroke handlers that run several
  // times before React re-renders, so the refs — not the state — are the source
  // of truth for "what is in the composer right now".
  const inputRef = useRef('')
  const tokensRef = useRef<ComposerToken[]>([])

  // The one collapsed paste the user expanded back into the text (Ctrl+P), so
  // the next Ctrl+P or the submit can fold the edited text back into a chip.
  const pasteSpanRef = useRef<null | PasteSpan>(null)
  const [pasteExpanded, setPasteExpanded] = useState(false)

  const setPasteSpan = useCallback((span: null | PasteSpan) => {
    pasteSpanRef.current = span
    setPasteExpanded(Boolean(span))
  }, [])

  const setInput = useCallback<StateSetter<string>>(next => {
    const prev = inputRef.current
    inputRef.current = typeof next === 'function' ? next(prev) : next

    if (!inputRef.current.startsWith(prev) || !inputRef.current) {
      draftLife.current++
    }

    if (inputRef.current !== landingValue.current) {
      bumpRevision()
    }

    landingValue.current = null

    if (pasteSpanRef.current) {
      setPasteSpan(shiftSpan(prev, inputRef.current, pasteSpanRef.current))
    }

    setInputState(inputRef.current)
  }, [bumpRevision, setPasteSpan])

  const setComposerTokens = useCallback<StateSetter<ComposerToken[]>>(next => {
    tokensRef.current = typeof next === 'function' ? next(tokensRef.current) : next
    setTokens(tokensRef.current)
  }, [])

  const isBlocked = useStore($isBlocked)
  const { querier } = useStdin() as { querier: Parameters<typeof readOsc52Clipboard>[0] }

  const {
    stage,
    queueRef,
    queueEditRef,
    queueDraft,
    queuedDisplay,
    queueEditIdx,
    enqueue,
    dequeue,
    prependQ,
    removeQ,
    setQueueEdit,
    takeQ
  } = useQueue(gw)

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, isBlocked, gw)

  const clearIn = useCallback(() => {
    draftLife.current++
    draftEpoch.current++
    setPasteSpan(null)
    setInput('')
    setInputBuf([])
    setComposerTokens([])
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }, [historyDraftRef, setComposerTokens, setHistoryIdx, setInput, setInputBuf, setPasteSpan, setQueueEdit])

  /**
   * Fold the expanded paste back into a chip; its payload is the text as
   * edited. A span that can't fold (another chip landed inside it) just stops
   * being a paste: its text stays in the draft as typed text.
   */
  const collapsePaste = useCallback(
    (value = inputRef.current) => {
      // The submitted value can run a keystroke ahead of the parent's copy.
      const span = pasteSpanRef.current && shiftSpan(inputRef.current, value, pasteSpanRef.current)
      setPasteSpan(null)
      const collapsed = span && collapsePasteSpan(value, span, tokensRef.current)

      if (!collapsed) {
        return value
      }

      setComposerTokens(collapsed.tokens)
      placeInputCursor(collapsed.value, collapsed.cursor)
      setInput(collapsed.value)

      return collapsed.value
    },
    [setComposerTokens, setInput, setPasteSpan]
  )

  /**
   * Ctrl+P in the composer: collapse the expanded paste, else expand the chip
   * under the cursor (or the last one). False when there was nothing to do.
   */
  const togglePaste = useCallback(
    (cursor: number) => {
      if (pasteSpanRef.current) {
        collapsePaste()

        return true
      }

      const expanded = expandPasteChip(inputRef.current, cursor, tokensRef.current)

      if (!expanded) {
        return false
      }

      setComposerTokens(expanded.tokens)
      placeInputCursor(expanded.value, expanded.cursor)
      setInput(expanded.value)
      setPasteSpan(expanded.span)

      return true
    },
    [collapsePaste, setComposerTokens, setInput, setPasteSpan]
  )

  /**
   * Deleting an `[[ Image N ]]` token IS how you unattach the image — there is
   * no separate control. Reconcile on every edit so the gateway's
   * `attached_images` never outlives the token the user just erased, which is
   * what used to make a stale image ride along on the next unrelated turn.
   */
  const syncTokens = useCallback(
    (value: string) => {
      // The draft is the buffered `\`+Enter lines plus the line being edited: a chip on an earlier
      // line is still in it. A sibling result landing first leaves this one's token textless for a moment.
      const own = value === landingValue.current

      const gone = droppedTokens(tokensRef.current, [...inputBufRef.current, value].join('\n')).filter(
        token => !(own && unlanded.current.has(token))
      )

      if (!gone.length) {
        return
      }

      for (const token of gone) {
        if (token.kind === 'image' && !gw.isCanonical) {
          void gw.request('image.detach', { path: token.path, session_id: getUiState().sid }).catch(() => {})
        }
      }

      gone.forEach(token => unlanded.current.delete(token))
      setComposerTokens(prev => prev.filter(token => !gone.includes(token)))
    },
    [gw, setComposerTokens]
  )

  const discardToken = useCallback(
    (token: ComposerToken) => {
      setComposerTokens(prev => prev.filter(t => t !== token))
      const shared = tokensRef.current.some(t => t.kind === 'image' && t.path === token.path)

      if (token.kind === 'image' && !shared && !gw.isCanonical) {
        void gw.request('image.detach', { path: token.path, session_id: getUiState().sid }).catch(() => {})
      }
    },
    [gw, setComposerTokens]
  )

  /**
   * A result inserting `text` at `cursor` of the draft it started on. Other
   * results may land on that draft first, so `land` re-places the insertion on
   * the composer's live value, after the insertions that landed since this
   * result started, and refuses once anything else edited the draft. The
   * caller writes the returned value straight away. `spaced` pads it off its
   * neighbours like a chip; raw text goes in verbatim.
   */
  const insertLanding = useCallback(
    (draft: DraftMark, value: string, cursor: number, text: string, spaced: boolean, token?: ComposerToken) => {
      const insert = (into: string, at: number) =>
        spaced ? insertAtCursor(into, at, text) : { cursor: at + text.length, value: into.slice(0, at) + text + into.slice(at) }

      if (token) {
        unlanded.current.add(token)
      }

      const land = (view?: ComposerView): ComposerPasteResult | null => {
        if (token) {
          unlanded.current.delete(token)
        }

        if (!isFresh(draft)) {
          // Nothing will show this chip now, so it must not ride along either.
          if (token) {
            discardToken(token)
          }

          return null
        }

        // The caller's view can lag the composer (a slash attach wrote it
        // before the input line re-rendered); the composer's value is the draft.
        const current = inputRef.current

        // After whatever landed at or before this point since this result
        // started: chips read in the order they were numbered, which is the
        // order their images reach the model. Landings its base already held
        // are part of `cursor`.
        const at = Math.min(
          current.length,
          landings.current
            .slice(draft.seen)
            .reduce((pos, prior) => (prior.at <= pos ? pos + prior.len : pos), cursor)
        )

        const next = insert(current, at)
        const len = next.value.length - current.length

        landings.current.push({ at, len, value: next.value })
        landingValue.current = next.value

        // The caret stays with the user's text when the insert landed behind it;
        // a caret measured on a stale view means nothing here, so it follows the insert.
        const caret = view && view.value === current ? view.caret : at

        return { cursor: Math.max(caret, at) + len, value: next.value }
      }

      return { ...insert(value, cursor), land }
    },
    [discardToken, isFresh]
  )

  /**
   * Attach an image the gateway already resolved: a token at the cursor,
   * followed by whatever non-path text came along with it (a drag-drop paste
   * of `~/shot.png look at this` keeps the caption).
   */
  const attachImageToken = useCallback(
    (
      draft: DraftMark,
      attached: ImageAttachResponse & { path?: string; mime?: string },
      value: string,
      cursor: number
    ): ComposerPasteResult => {
      const index = nextImageIndex(tokensRef.current)
      const label = imageToken(index)

      const token: ComposerToken = {
        index,
        kind: 'image',
        label,
        path: attached.path ?? '',
        ...(attached.mime ? { mime: attached.mime } : {})
      }

      setComposerTokens(prev => trimTokens([...prev, token]))

      const remainder = attached.remainder?.trim() ?? ''

      return insertLanding(draft, value, cursor, remainder ? `${label} ${remainder}` : label, true, token)
    },
    [insertLanding, setComposerTokens]
  )

  const attachmentFlights = useRef(0)
  const staleAttachments = useRef<Array<{ destination: SubmissionDestination; path: string }>>([])

  const resolveAttachment = useCallback(
    async <T extends { path?: string }>(
      destination: SubmissionDestination,
      draft: DraftMark,
      request: Promise<T | null>,
      accept: (attached: T | null) => ComposerPasteResult | null,
      fresh = () => isFresh(draft)
    ): Promise<ComposerPasteResult | null> => {
      attachmentFlights.current++

      try {
        const attached = await request

        if (isCurrentDestination(destination) && fresh()) {
          return accept(attached)
        }

        if (attached?.path && !gw.isCanonical) {
          staleAttachments.current.push({ destination, path: attached.path })
        }

        return null
      } finally {
        attachmentFlights.current--

        // Detach is path-based: wait for concurrent replies before deciding
        // whether that path belongs to a surviving visible attachment.
        if (!attachmentFlights.current) {
          for (const stale of staleAttachments.current.splice(0)) {
            if (
              isCurrentDestination(stale.destination) &&
              tokensRef.current.some(token => token.kind === 'image' && token.path === stale.path)
            ) {
              continue
            }

            void gw.request('image.detach', { session_id: stale.destination.sid, path: stale.path }).catch(() => {})
          }
        }
      }
    },
    [gw, isFresh]
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
      const destination = captureDestination()
      const draft = captureDraft(value)
      const { sid } = destination

      if (!sid) {
        return null
      }

      return resolveAttachment<ClipboardPasteResponse & { path?: string; mime?: string }>(
        destination,
        draft,
        (gw.isCanonical
          ? stageClipboardImage(gw, destination).then(image => (image ? { ...image, attached: true } : null))
          : gw.request<ClipboardPasteResponse & { path?: string; mime?: string }>('clipboard.paste', {
              session_id: sid
            })
        ).catch((error: Error) => {
          if (!quiet) {
            sys(t('canonical.queue.clipboardImageFailed', error.message))
          }

          return null
        }),
        r => {
          if (r?.attached) {
            return attachImageToken(draft, r, value, cursor)
          }

          if (!quiet) {
            sys(r?.message || 'No image found in clipboard')
          }

          return null
        }
      )
    },
    [attachImageToken, captureDraft, gw, resolveAttachment, sys]
  )

  const handleResolvedPaste = useCallback(
    async ({ bracketed, cursor, text, value }: Omit<PasteEvent, 'hotkey'>): Promise<ComposerPasteResult | null> => {
      const destination = captureDestination()
      const draft = captureDraft(value)
      const current = () => isCurrentDestination(destination) && isFresh(draft)
      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText || !/[^\n]/.test(cleanedText)) {
        return bracketed ? pasteClipboardImage(value, cursor, true) : null
      }

      const sid = getUiState().sid

      if (sid && looksLikeDroppedPath(cleanedText)) {
        try {
          const next = await resolveAttachment<ImageAttachResponse & { path?: string; mime?: string }>(
            destination,
            draft,
            gw.isCanonical
              ? stageImagePath(cleanedText, gw, destination)
              : gw.request<ImageAttachResponse & { path?: string }>('image.attach', {
                  path: cleanedText,
                  session_id: sid
                }),
            attached => (attached?.name ? attachImageToken(draft, attached, value, cursor) : null)
          )

          if (next) {
            return next
          }
        } catch {
          // Fall back to generic file-drop detection below.
        }

        if (!current()) {
          return null
        }

        try {
          const dropped = await gw.request<InputDetectDropResponse>('input.detect_drop', {
            session_id: sid,
            text: cleanedText
          })

          if (!current()) {
            return null
          }

          if (dropped?.matched && dropped.text) {
            return insertLanding(draft, value, cursor, dropped.text, true)
          }
        } catch {
          // Fall through to normal text paste behavior.
        }
      }

      if (!current()) {
        return null
      }

      const lineCount = cleanedText.split('\n').length

      // Inside an expanded paste the text joins it: a chip there would fold
      // into the outer paste as its label, not its content.
      if (insideSpan(pasteSpanRef.current, cursor) || !shouldCollapsePaste(cleanedText, lineCount)) {
        return insertLanding(draft, value, cursor, cleanedText, false)
      }

      const label = pasteTokenLabel(cleanedText, lineCount)
      const token: ComposerToken = { kind: 'paste', label, text: cleanedText }
      setComposerTokens(prev => trimTokens([...prev, token]))
      const inserted = insertLanding(draft, value, cursor, label, true, token)

      void gw
        .request<{ path?: string }>('paste.collapse', { text: cleanedText })
        .then(r => {
          const path = r?.path

          if (!path || !isCurrentDestination(destination)) {
            return
          }

          setComposerTokens(prev => prev.map(t => (t === token ? { ...t, path } : t)))
        })
        .catch(() => {})

      return inserted
    },
    [attachImageToken, captureDraft, gw, insertLanding, isFresh, pasteClipboardImage, resolveAttachment, setComposerTokens]
  )

  const handleTextPaste = useCallback(
    ({ bracketed, cursor, hotkey, text, value }: PasteEvent): MaybePromise<ComposerPasteResult | null> => {
      // Clipboard reads can outlive a focus change, before any gateway RPC starts.
      const destination = captureDestination()
      const draft = captureDraft(value)

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
          if (!(isCurrentDestination(destination) && isFresh(draft))) {
            return null
          }

          if (isUsableClipboardText(preferredText)) {
            return handleResolvedPaste({ bracketed: false, cursor, text: preferredText, value })
          }

          // No text on the clipboard — an image paste looks exactly like this.
          return pasteClipboardImage(value, cursor, false)
        })
      }

      return handleResolvedPaste({ bracketed: !!bracketed, cursor, text, value })
    },
    [captureDraft, handleResolvedPaste, isFresh, pasteClipboardImage, querier]
  )

  /**
   * `/paste` and `/image` attach without a cursor of their own — the token
   * lands at the end of whatever is currently typed.
   */
  const appendAttachment = useCallback(
    (attach: (value: string, cursor: number) => Promise<ComposerPasteResult | null>) => {
      const destination = captureDestination()
      const current = inputRef.current

      // An attach lands on the live draft, after any sibling paste that landed first; one the user
      // edited away meanwhile is discarded as stale.
      return attach(current, current.length).then(next => {
        const landed = next && isCurrentDestination(destination) && next.land?.()

        if (landed) {
          setInput(landed.value)
        }
      })
    },
    [setInput]
  )

  const attachClipboardImage = useCallback(
    () => appendAttachment((value, cursor) => pasteClipboardImage(value, cursor, false)),
    [appendAttachment, pasteClipboardImage]
  )

  /**
   * Attach images by path, one after another, each landing at the end of the draft. The destination
   * is captured ONCE and every attach targets it. With `followTyping` (an edited prompt's restore),
   * typing on keeps it going and the image lands after the typed text; otherwise any other edit, like
   * a session switch or the draft cleared, submitted or replaced, drops the rest, and a restore says
   * so when that leaves images behind in the same session.
   * Sequential because a second attach started before the first lands would see it as foreign.
   */
  const attachImagePaths = useCallback(
    async (paths: readonly string[], { followTyping = false } = {}) => {
      const destination = captureDestination()
      const life = draftLife.current
      let draft = captureDraft(inputRef.current)

      // Unchanged since our last write or, following typing, the same draft only added to at the end.
      const current = () =>
        isCurrentDestination(destination) &&
        (isFresh(draft) || (followTyping && life === draftLife.current))

      const dropped = (count: number) => followTyping && isCurrentDestination(destination) && sys(t('canonical.queue.imagesNotRestored', count))

      for (const [done, path] of paths.entries()) {
        if (!destination.sid || !path.trim()) {
          return
        }

        if (!current()) {
          return void dropped(paths.length - done)
        }

        const next = await resolveAttachment<ImageAttachResponse & { path?: string; mime?: string }>(
          destination,
          draft,
          (gw.isCanonical
            ? stageImagePath(path, gw, destination)
            : gw.request<ImageAttachResponse & { path?: string }>('image.attach', {
                path,
                session_id: destination.sid
              })
          ).catch((e: Error) => {
            if (current()) {
              sys(`error: ${e.message}`)
            }

            return null
          }),
          // At the live end: typing a followed restore kept, or a sibling paste that landed first.
          attached =>
            attached?.name
              ? attachImageToken(captureDraft(inputRef.current), attached, inputRef.current, inputRef.current.length)
              : null,
          current
        )

        // Re-placed after any paste that landed between the accept and here; null when edited away,
        // which loses this image even while the draft is still live.
        const landed = next?.land?.()

        if (!landed) {
          return void ((!next && current()) || dropped(paths.length - done))
        }

        setInput(landed.value)
        draft = captureDraft(inputRef.current)
      }
    },
    [attachImageToken, captureDraft, gw, isFresh, resolveAttachment, setInput, sys]
  )

  // Text from outside the composer (backtrack browse) lands in arrival order: keys typed behind a
  // paste in the same terminal read wait for the paste to resolve.
  const pendingAppend = useRef<null | Promise<void>>(null)

  // The draft a browse key or paste was typed into: its session and its epoch. A paste resolving
  // (image or file lookup) after a session switch, or after the draft was cleared or submitted, is
  // dropped with the keys queued behind it; it never lands in the next draft.
  const liveDraft = useCallback(() => {
    const destination = captureDestination()
    const epoch = draftEpoch.current

    return () => isCurrentDestination(destination) && epoch === draftEpoch.current
  }, [])

  /** A key typed outside the composer (backtrack browse), appended after any paste still resolving. */
  const appendText = useCallback(
    (text: string) => {
      if (!pendingAppend.current) {
        return setInput(prev => prev + text)
      }

      const live = liveDraft()

      pendingAppend.current = pendingAppend.current.then(() => {
        if (live()) {
          setInput(prev => prev + text)
        }
      })
    },
    [liveDraft, setInput]
  )

  /** Text pasted outside the composer (backtrack browse), handled as a composer paste: collapse,
   * dropped-path and image detection included, at the end of the draft. If the user typed on in the
   * same draft while it resolved, the raw text is appended instead, as the composer's own paste does.
   * If the draft itself went away (session switch, cleared, submitted), the paste is
   * dropped with a notice, never landing in the next draft. */
  const appendPaste = useCallback(
    (text: string) => {
      const live = liveDraft()

      const run = async () => {
        const value = inputRef.current
        const revision = composerRevision.current
        const result = live() ? await handleTextPaste({ bracketed: true, cursor: value.length, text, value }) : null

        if (!live()) {
          if (text.trim()) {
            sys(t('canonical.queue.pasteNotApplied'))
          }

          return
        }

        // A restore attach can land while this resolves; the result re-places itself after it.
        const landed = result?.land ? result.land() : result && revision === composerRevision.current ? result : null
        const next = landed ? landed.value : inputRef.current + text

        if (landed || text.trim()) {
          placeInputCursor(next, next.length)
          setInput(next)
        }
      }

      const done = (pendingAppend.current ?? Promise.resolve()).then(run)

      const tail = done.finally(() => {
        if (pendingAppend.current === tail) {
          pendingAppend.current = null
        }
      })

      pendingAppend.current = tail

      return done
    },
    [handleTextPaste, liveDraft, setInput, sys]
  )

  /** Put a submitted draft back as it was (text with its chips, and their tokens) when a send that
   * took it out of the composer cut nothing. Only into the session it was typed in; a draft the user
   * started since stays, after it, and an image label both use is renumbered on the restored side. */
  const restoreDraft = useCallback(
    (display: string, draftTokens: readonly ComposerToken[], destination: SubmissionDestination) => {
      if (!isCurrentDestination(destination)) {
        return false
      }

      let restored = display
      const kept: ComposerToken[] = []

      for (const token of draftTokens) {
        if (token.kind !== 'image' || !tokensRef.current.some(t => t.label === token.label)) {
          kept.push(token)

          continue
        }

        const index = nextImageIndex([...tokensRef.current, ...draftTokens, ...kept])
        const label = imageToken(index)

        restored = restored.replace(token.label, label)
        kept.push({ ...token, index, label })
      }

      const current = inputRef.current
      const buffered = inputBufRef.current.length > 0

      setComposerTokens(prev => [...kept, ...prev])

      if (buffered) {
        setInputBuf(prev => [restored, ...prev])
      } else {
        const next = current ? `${restored}\n${current}` : restored

        placeInputCursor(next, next.length)
        setInput(next)
      }

      return true
    },
    [setComposerTokens, setInput, setInputBuf]
  )

  const attachImagePath = useCallback((path: string) => attachImagePaths([path]), [attachImagePaths])

  const openEditor = useCallback(async () => {
    const destination = captureDestination()
    const draft = captureDraft(inputRef.current)
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

      if (!(isCurrentDestination(destination) && isFresh(draft))) {
        enqueue(text, text, destination)

        return
      }

      setInput('')
      setInputBuf([])
      submitRef.current(text)
    } finally {
      rmSync(dir, { force: true, recursive: true })
    }
  }, [captureDraft, enqueue, input, inputBuf, isFresh, setInput, setInputBuf, submitRef])

  const actions = useMemo(
    () => ({
      attachClipboardImage,
      attachImagePath,
      appendPaste,
      appendText,
      attachImagePaths,
      clearIn,
      collapsePaste,
      stage,
      dequeue,
      enqueue,
      handleTextPaste,
      openEditor,
      prependQueue: prependQ,
      restoreDraft,
      pushHistory,
      queueDraft,
      removeQueue: removeQ,
      setCompIdx,
      setComposerTokens,
      setHistoryIdx,
      setInput,
      setInputBuf,
      setQueueEdit,
      takeQueue: takeQ,
      syncTokens,
      togglePaste
    }),
    [
      attachClipboardImage,
      attachImagePath,
      appendPaste,
      appendText,
      attachImagePaths,
      clearIn,
      stage,
      dequeue,
      enqueue,
      handleTextPaste,
      openEditor,
      prependQ,
      restoreDraft,
      pushHistory,
      queueDraft,
      removeQ,
      setCompIdx,
      setComposerTokens,
      setHistoryIdx,
      setInput,
      setQueueEdit,
      takeQ,
      setInputBuf,
      syncTokens,
      collapsePaste,
      togglePaste
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
      pasteExpanded,
      queueEditIdx,
      queuedDisplay,
      tokens
    }),
    [compIdx, compReplace, completions, historyIdx, input, inputBuf, pasteExpanded, queueEditIdx, queuedDisplay, tokens]
  )

  return {
    actions,
    refs,
    state
  }
}
