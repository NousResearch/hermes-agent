import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { useStore } from '@nanostores/react'
import { useCallback, useMemo, useState } from 'react'
import { useStdin } from '@hermes/ink'

import type { PasteEvent } from '../components/textInput.js'
import { LARGE_PASTE } from '../config/limits.js'
import { useCompletion } from '../hooks/useCompletion.js'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { useQueue } from '../hooks/useQueue.js'
import { isUsableClipboardText, readClipboardText } from '../lib/clipboard.js'
import { readOsc52Clipboard } from '../lib/osc52.js'
import { pasteTokenLabel, stripTrailingPasteNewlines } from '../lib/text.js'
import type { InputDetectDropResponse } from '../gatewayTypes.js'

import type { MaybePromise, PasteSnippet, UseComposerStateOptions, UseComposerStateResult } from './interfaces.js'
import { $isBlocked } from './overlayStore.js'
import { getUiState } from './uiStore.js'

const PASTE_SNIP_MAX_COUNT = 32
const PASTE_SNIP_MAX_TOTAL_BYTES = 4 * 1024 * 1024

const trimSnips = (snips: PasteSnippet[]): PasteSnippet[] => {
  let total = 0
  const out: PasteSnippet[] = []

  for (let i = snips.length - 1; i >= 0; i--) {
    const snip = snips[i]!
    const size = snip.text.length

    if (out.length >= PASTE_SNIP_MAX_COUNT || total + size > PASTE_SNIP_MAX_TOTAL_BYTES) {
      break
    }

    total += size
    out.unshift(snip)
  }

  return out.length === snips.length ? snips : out
}

export function looksLikeDroppedPath(text: string): boolean {
  const trimmed = text.trim()

  if (!trimmed || trimmed.includes('\n')) {
    return false
  }

  return (
    trimmed.startsWith('/') ||
    trimmed.startsWith('~') ||
    trimmed.startsWith('./') ||
    trimmed.startsWith('../') ||
    trimmed.startsWith('file://') ||
    trimmed.startsWith('"/') ||
    trimmed.startsWith("'/") ||
    trimmed.startsWith('"~') ||
    trimmed.startsWith("'~") ||
    (/^[A-Za-z]:[\\/]/.test(trimmed)) ||
    (/^["'][A-Za-z]:[\\/]/.test(trimmed))
  )
}

export function useComposerState({ gw, onClipboardPaste, onImageAttached, submitRef }: UseComposerStateOptions): UseComposerStateResult {
  const [input, setInput] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [pasteSnips, setPasteSnips] = useState<PasteSnippet[]>([])
  const isBlocked = useStore($isBlocked)
  const { querier } = useStdin() as { querier: Parameters<typeof readOsc52Clipboard>[0] }

  const { queueRef, queueEditRef, queuedDisplay, queueEditIdx, enqueue, dequeue, replaceQ, setQueueEdit, syncQueue } =
    useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, isBlocked, gw)

  const clearIn = useCallback(() => {
    setInput('')
    setInputBuf([])
    setPasteSnips([])
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }, [historyDraftRef, setQueueEdit, setHistoryIdx])

  const handleResolvedPaste = useCallback(
    async ({ bracketed, cursor, text, value }: Omit<PasteEvent, 'hotkey'>): Promise<null | { cursor: number; value: string }> => {
      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText || !/[^\n]/.test(cleanedText)) {
        if (bracketed) {
          void onClipboardPaste(true)
        }

        return null
      }

      const sid = getUiState().sid
      if (sid && looksLikeDroppedPath(cleanedText)) {
        try {
          const attached = await gw.request<InputDetectDropResponse & { remainder?: string }>('image.attach', {
            path: cleanedText,
            session_id: sid
          })

          if (attached?.name) {
            onImageAttached?.(attached)
            const remainder = attached.remainder?.trim() ?? ''
            if (!remainder) {
              return { cursor, value }
            }

            const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
            const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
            const insert = `${lead}${remainder}${tail}`

            return {
              cursor: cursor + insert.length,
              value: value.slice(0, cursor) + insert + value.slice(cursor)
            }
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
            const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
            const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
            const insert = `${lead}${dropped.text}${tail}`

            return {
              cursor: cursor + insert.length,
              value: value.slice(0, cursor) + insert + value.slice(cursor)
            }
          }
        } catch {
          // Fall through to normal text paste behavior.
        }
      }

      const lineCount = cleanedText.split('\n').length

      if (cleanedText.length < LARGE_PASTE.chars && lineCount < LARGE_PASTE.lines) {
        return {
          cursor: cursor + cleanedText.length,
          value: value.slice(0, cursor) + cleanedText + value.slice(cursor)
        }
      }

      const label = pasteTokenLabel(cleanedText, lineCount)
      const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
      const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
      const insert = `${lead}${label}${tail}`

      setPasteSnips(prev => trimSnips([...prev, { label, text: cleanedText }]))

      void gw
        .request<{ path?: string }>('paste.collapse', { text: cleanedText })
        .then(r => {
          const path = r?.path

          if (!path) {
            return
          }

          setPasteSnips(prev => prev.map(s => (s.label === label ? { ...s, path } : s)))
        })
        .catch(() => {})

      return {
        cursor: cursor + insert.length,
        value: value.slice(0, cursor) + insert + value.slice(cursor)
      }
    },
    [gw, onClipboardPaste, onImageAttached]
  )

  const handleTextPaste = useCallback(
    ({ bracketed, cursor, hotkey, text, value }: PasteEvent): MaybePromise<null | { cursor: number; value: string }> => {
      if (hotkey) {
        const preferOsc52 = Boolean(process.env.SSH_CONNECTION || process.env.SSH_TTY || process.env.SSH_CLIENT)
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

          void onClipboardPaste(false)
          return null
        })
      }

      return handleResolvedPaste({ bracketed: !!bracketed, cursor, text, value })
    },
    [gw, handleResolvedPaste, onClipboardPaste, querier]
  )

  const openEditor = useCallback(() => {
    const editor = process.env.EDITOR || process.env.VISUAL || 'vi'
    const file = join(mkdtempSync(join(tmpdir(), 'hermes-')), 'prompt.md')

    writeFileSync(file, [...inputBuf, input].join('\n'))
    process.stdout.write('\x1b[?1049l')
    const { status: code } = spawnSync(editor, [file], { stdio: 'inherit' })
    process.stdout.write('\x1b[?1049h\x1b[2J\x1b[H')

    if (code === 0) {
      const text = readFileSync(file, 'utf8').trimEnd()

      if (text) {
        setInput('')
        setInputBuf([])
        submitRef.current(text)
      }
    }

    rmSync(file, { force: true })
  }, [input, inputBuf, submitRef])

  const actions = useMemo(
    () => ({
      clearIn,
      dequeue,
      enqueue,
      handleTextPaste,
      openEditor,
      pushHistory,
      replaceQueue: replaceQ,
      setCompIdx,
      setHistoryIdx,
      setInput,
      setInputBuf,
      setPasteSnips,
      setQueueEdit,
      syncQueue
    }),
    [
      clearIn,
      dequeue,
      enqueue,
      handleTextPaste,
      openEditor,
      pushHistory,
      replaceQ,
      setCompIdx,
      setHistoryIdx,
      setQueueEdit,
      syncQueue
    ]
  )

  const refs = useMemo(
    () => ({
      historyDraftRef,
      historyRef,
      queueEditRef,
      queueRef,
      submitRef
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
      pasteSnips,
      queueEditIdx,
      queuedDisplay
    }),
    [compIdx, compReplace, completions, historyIdx, input, inputBuf, pasteSnips, queueEditIdx, queuedDisplay]
  )

  return {
    actions,
    refs,
    state
  }
}
