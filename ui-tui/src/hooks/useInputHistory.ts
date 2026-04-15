import { useCallback, useRef, useState } from 'react'

import * as inputHistory from '../lib/history.js'

export function useInputHistory() {
  const historyRef = useRef<string[]>(inputHistory.load())
  const [historyIdx, setHistoryIdx] = useState<number | null>(null)
  const historyDraftRef = useRef('')

  const pushHistory = useCallback((text: string) => {
    inputHistory.append(text)
  }, [])

  return { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory }
}
