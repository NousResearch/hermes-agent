import { useEffect, useRef } from 'react';
import { resolveDetailsMode } from '../domain/details.js';
import { asRpcResult } from '../lib/rpc.js';
import { turnController } from './turnController.js';
import { patchUiState } from './uiStore.js';
const MTIME_POLL_MS = 5000;
const quietRpc = async (gw, method, params = {}) => {
    try {
        return asRpcResult(await gw.request(method, params));
    }
    catch {
        return null;
    }
};
export const applyDisplay = (cfg, setBell) => {
    const d = cfg?.config?.display ?? {};
    setBell(!!d.bell_on_complete);
    patchUiState({
        compact: !!d.tui_compact,
        detailsMode: resolveDetailsMode(d),
        inlineDiffs: d.inline_diffs !== false,
        showCost: !!d.show_cost,
        showReasoning: !!d.show_reasoning,
        statusBar: d.tui_statusbar !== false,
        streaming: d.streaming !== false
    });
};
export function useConfigSync({ gw, setBellOnComplete, setVoiceEnabled, sid }) {
    const mtimeRef = useRef(0);
    useEffect(() => {
        if (!sid) {
            return;
        }
        quietRpc(gw, 'voice.toggle', { action: 'status' }).then(r => setVoiceEnabled(!!r?.enabled));
        quietRpc(gw, 'config.get', { key: 'mtime' }).then(r => {
            mtimeRef.current = Number(r?.mtime ?? 0);
        });
        quietRpc(gw, 'config.get', { key: 'full' }).then(r => applyDisplay(r, setBellOnComplete));
    }, [gw, setBellOnComplete, setVoiceEnabled, sid]);
    useEffect(() => {
        if (!sid) {
            return;
        }
        const id = setInterval(() => {
            quietRpc(gw, 'config.get', { key: 'mtime' }).then(r => {
                const next = Number(r?.mtime ?? 0);
                if (!mtimeRef.current) {
                    if (next) {
                        mtimeRef.current = next;
                    }
                    return;
                }
                if (!next || next === mtimeRef.current) {
                    return;
                }
                mtimeRef.current = next;
                quietRpc(gw, 'reload.mcp', { session_id: sid }).then(r => r && turnController.pushActivity('MCP reloaded after config change'));
                quietRpc(gw, 'config.get', { key: 'full' }).then(r => applyDisplay(r, setBellOnComplete));
            });
        }, MTIME_POLL_MS);
        return () => clearInterval(id);
    }, [gw, setBellOnComplete, sid]);
}
