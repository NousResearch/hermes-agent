import { useRef, useState } from 'react';
import * as inputHistory from '../lib/history.js';
export function useInputHistory() {
    const historyRef = useRef(inputHistory.load());
    const [historyIdx, setHistoryIdx] = useState(null);
    const historyDraftRef = useRef('');
    return { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory: inputHistory.append };
}
