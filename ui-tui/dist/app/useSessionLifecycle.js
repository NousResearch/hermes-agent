import { useCallback } from 'react';
import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js';
import { introMsg, toTranscriptMessages } from '../domain/messages.js';
import { ZERO } from '../domain/usage.js';
import { asRpcResult } from '../lib/rpc.js';
import { patchOverlayState } from './overlayStore.js';
import { turnController } from './turnController.js';
import { patchTurnState } from './turnStore.js';
import { getUiState, patchUiState } from './uiStore.js';
const usageFrom = (info) => (info?.usage ? { ...ZERO, ...info.usage } : ZERO);
const trimTail = (items) => {
    const q = [...items];
    while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') {
        q.pop();
    }
    if (q.at(-1)?.role === 'user') {
        q.pop();
    }
    return q;
};
export function useSessionLifecycle(opts) {
    const { colsRef, composerActions, gw, panel, rpc, scrollRef, setHistoryItems, setLastUserMsg, setSessionStartedAt, setStickyPrompt, setVoiceProcessing, setVoiceRecording, sys } = opts;
    const closeSession = useCallback((targetSid) => targetSid ? rpc('session.close', { session_id: targetSid }) : Promise.resolve(null), [rpc]);
    const resetSession = useCallback(() => {
        turnController.fullReset();
        setVoiceRecording(false);
        setVoiceProcessing(false);
        patchUiState({ bgTasks: new Set(), info: null, sid: null, usage: ZERO });
        setHistoryItems([]);
        setLastUserMsg('');
        setStickyPrompt('');
        composerActions.setPasteSnips([]);
    }, [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt, setVoiceProcessing, setVoiceRecording]);
    const resetVisibleHistory = useCallback((info = null) => {
        turnController.idle();
        turnController.clearReasoning();
        turnController.turnTools = [];
        turnController.persistedToolLabels.clear();
        setHistoryItems(info ? [introMsg(info)] : []);
        setStickyPrompt('');
        setLastUserMsg('');
        composerActions.setPasteSnips([]);
        patchTurnState({ activity: [] });
        patchUiState({ info, usage: usageFrom(info) });
    }, [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt]);
    const newSession = useCallback(async (msg) => {
        const setup = await rpc('setup.status', {});
        if (setup?.provider_configured === false) {
            panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections());
            patchUiState({ status: 'setup required' });
            return;
        }
        await closeSession(getUiState().sid);
        const r = await rpc('session.create', { cols: colsRef.current });
        if (!r) {
            return patchUiState({ status: 'ready' });
        }
        const info = r.info ?? null;
        resetSession();
        setSessionStartedAt(Date.now());
        patchUiState({
            info,
            sid: r.session_id,
            status: info?.version ? 'ready' : 'starting agent…',
            usage: usageFrom(info)
        });
        if (info) {
            setHistoryItems([introMsg(info)]);
        }
        if (info?.credential_warning) {
            sys(`warning: ${info.credential_warning}`);
        }
        if (msg) {
            sys(msg);
        }
    }, [closeSession, colsRef, panel, resetSession, rpc, setHistoryItems, setSessionStartedAt, sys]);
    const resumeById = useCallback((id) => {
        patchOverlayState({ picker: false });
        patchUiState({ status: 'resuming…' });
        rpc('setup.status', {}).then(setup => {
            if (setup?.provider_configured === false) {
                panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections());
                patchUiState({ status: 'setup required' });
                return;
            }
            closeSession(getUiState().sid === id ? null : getUiState().sid).then(() => gw
                .request('session.resume', { cols: colsRef.current, session_id: id })
                .then(raw => {
                const r = asRpcResult(raw);
                if (!r) {
                    sys('error: invalid response: session.resume');
                    return patchUiState({ status: 'ready' });
                }
                resetSession();
                setSessionStartedAt(Date.now());
                const resumed = toTranscriptMessages(r.messages);
                setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed);
                patchUiState({
                    info: r.info ?? null,
                    sid: r.session_id,
                    status: 'ready',
                    usage: usageFrom(r.info ?? null)
                });
                setTimeout(() => scrollRef.current?.scrollToBottom(), 0);
            })
                .catch((e) => {
                sys(`error: ${e.message}`);
                patchUiState({ status: 'ready' });
            }));
        });
    }, [closeSession, colsRef, gw, panel, resetSession, rpc, scrollRef, setHistoryItems, setSessionStartedAt, sys]);
    const guardBusySessionSwitch = useCallback((what = 'switch sessions') => {
        if (!getUiState().busy) {
            return false;
        }
        sys(`interrupt the current turn before trying to ${what}`);
        return true;
    }, [sys]);
    return {
        closeSession,
        guardBusySessionSwitch,
        newSession,
        resetSession,
        resetVisibleHistory,
        resumeById,
        trimLastExchange: trimTail
    };
}
