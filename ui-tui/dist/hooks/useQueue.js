import { useCallback, useRef, useState } from 'react';
export function useQueue() {
    const queueRef = useRef([]);
    const [queuedDisplay, setQueuedDisplay] = useState([]);
    const queueEditRef = useRef(null);
    const [queueEditIdx, setQueueEditIdx] = useState(null);
    const syncQueue = useCallback(() => setQueuedDisplay([...queueRef.current]), []);
    const setQueueEdit = useCallback((idx) => {
        queueEditRef.current = idx;
        setQueueEditIdx(idx);
    }, []);
    const enqueue = useCallback((text) => {
        queueRef.current.push(text);
        syncQueue();
    }, [syncQueue]);
    const dequeue = useCallback(() => {
        const head = queueRef.current.shift();
        syncQueue();
        return head;
    }, [syncQueue]);
    const replaceQ = useCallback((i, text) => {
        queueRef.current[i] = text;
        syncQueue();
    }, [syncQueue]);
    return {
        dequeue,
        enqueue,
        queueEditIdx,
        queueEditRef,
        queueRef,
        queuedDisplay,
        replaceQ,
        setQueueEdit,
        syncQueue
    };
}
