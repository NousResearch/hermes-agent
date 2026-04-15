import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { useStore } from '@nanostores/react'
import { type Dispatch, type MutableRefObject, type SetStateAction, useCallback, useState } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import type { GatewayClient } from '../gatewayClient.js'
import { useCompletion } from '../hooks/useCompletion.js'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { useQueue } from '../hooks/useQueue.js'
import { pasteTokenLabel, stripTrailingPasteNewlines } from '../lib/text.js'

import { LARGE_PASTE } from './constants.js'
import type { PasteSnippet } from './helpers.js'
import type { CompletionItem } from './interfaces.js'
import { $isBlocked } from './overlayStore.js'

export interface ComposerPasteResult {
  cursor: number
  value: string
}

export interface ComposerActions {
  clearIn: () => void
  dequeue: () => string | undefined
  enqueue: (text: string) => void
  handleTextPaste: (event: PasteEvent) => ComposerPasteResult | null
  openEditor: () => void
  pushHistory: (text: string) => void
  replaceQueue: (index: number, text: string) => void
  setCompIdx: Dispatch<SetStateAction<number>>
  setHistoryIdx: Dispatch<SetStateAction<number | null>>
  setInput: Dispatch<SetStateAction<string>>
  setInputBuf: Dispatch<SetStateAction<string[]>>
  setPasteSnips: Dispatch<SetStateAction<PasteSnippet[]>>
  setQueueEdit: (index: number | null) => void
  syncQueue: () => void
}

export interface ComposerRefs {
  historyDraftRef: MutableRefObject<string>
  historyRef: MutableRefObject<string[]>
  queueEditRef: MutableRefObject<number | null>
  queueRef: MutableRefObject<string[]>
  submitRef: MutableRefObject<(value: string) => void>
}

export interface ComposerState {
  compIdx: number
  compReplace: number
  completions: CompletionItem[]
  historyIdx: number | null
  input: string
  inputBuf: string[]
  pasteSnips: PasteSnippet[]
  queueEditIdx: number | null
  queuedDisplay: string[]
}

export interface UseComposerStateOptions {
  gw: GatewayClient
  onClipboardPaste: (quiet?: boolean) => Promise<void> | void
  submitRef: MutableRefObject<(value: string) => void>
}

export interface UseComposerStateResult {
  actions: ComposerActions
  refs: ComposerRefs
  state: ComposerState
}

export function useComposerState({ gw, onClipboardPaste, submitRef }: UseComposerStateOptions): UseComposerStateResult {
  const [input, setInput] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [pasteSnips, setPasteSnips] = useState<PasteSnippet[]>([])
  const isBlocked = useStore($isBlocked)

  const { queueRef, queueEditRef, queuedDisplay, queueEditIdx, enqueue, dequeue, replaceQ, setQueueEdit, syncQueue } =
    useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, isBlocked, gw)

  const clearIn = useCallback(() => {
    setInput('')
    setInputBuf([])
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }, [historyDraftRef, setQueueEdit, setHistoryIdx])

  const handleTextPaste = useCallback(
    ({ bracketed, cursor, hotkey, text, value }: PasteEvent) => {
      if (hotkey) {
        void onClipboardPaste(false)

        return null
      }

      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText || !/[^\n]/.test(cleanedText)) {
        if (bracketed) {
          void onClipboardPaste(true)
        }

        return null
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

      setPasteSnips(prev => [...prev, { label, text: cleanedText }].slice(-32))

      return {
        cursor: cursor + insert.length,
        value: value.slice(0, cursor) + insert + value.slice(cursor)
      }
    },
    [onClipboardPaste]
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

    try {
      unlinkSync(file)
    } catch {
      /* noop */
    }
  }, [input, inputBuf, submitRef])

  return {
    actions: {
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
    },
    refs: {
      historyDraftRef,
      historyRef,
      queueEditRef,
      queueRef,
      submitRef
    },
    state: {
      compIdx,
      compReplace,
      completions,
      historyIdx,
      input,
      inputBuf,
      pasteSnips,
      queueEditIdx,
      queuedDisplay
    }
  }
}
