import { useRef, useState } from 'react'

import * as inputHistory from '../lib/history.js'

export function useInputHistory() {
  const historyRef = useRef<string[]>(inputHistory.load())
  const [historyIdx, setHistoryIdx] = useState<number | null>(null)
  const historyDraftRef = useRef('')

  const pushHistory = (text: string) => {
    const trimmed = text.trim()

    if (trimmed && historyRef.current.at(-1) !== trimmed) {
      historyRef.current.push(trimmed)
      inputHistory.append(trimmed)
    }
  }

  return { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory }
}
