import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Text, useInput } from '@hermes/ink';
import { useEffect, useMemo, useState } from 'react';
import { providerDisplayNames } from '../domain/providers.js';
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js';
const VISIBLE = 12;
const pageOffset = (count, sel) => Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), count - VISIBLE));
const visibleItems = (items, sel) => {
    const off = pageOffset(items.length, sel);
    return { items: items.slice(off, off + VISIBLE), off };
};
export function ModelPicker({ gw, onCancel, onSelect, sessionId, t }) {
    const [providers, setProviders] = useState([]);
    const [currentModel, setCurrentModel] = useState('');
    const [err, setErr] = useState('');
    const [loading, setLoading] = useState(true);
    const [persistGlobal, setPersistGlobal] = useState(false);
    const [providerIdx, setProviderIdx] = useState(0);
    const [modelIdx, setModelIdx] = useState(0);
    const [stage, setStage] = useState('provider');
    useEffect(() => {
        gw.request('model.options', sessionId ? { session_id: sessionId } : {})
            .then(raw => {
            const r = asRpcResult(raw);
            if (!r) {
                setErr('invalid response: model.options');
                setLoading(false);
                return;
            }
            const next = r.providers ?? [];
            setProviders(next);
            setCurrentModel(String(r.model ?? ''));
            setProviderIdx(Math.max(0, next.findIndex(p => p.is_current)));
            setModelIdx(0);
            setErr('');
            setLoading(false);
        })
            .catch((e) => {
            setErr(rpcErrorMessage(e));
            setLoading(false);
        });
    }, [gw, sessionId]);
    const provider = providers[providerIdx];
    const models = provider?.models ?? [];
    const names = useMemo(() => providerDisplayNames(providers), [providers]);
    useInput((ch, key) => {
        if (key.escape) {
            if (stage === 'model') {
                setStage('provider');
                setModelIdx(0);
                return;
            }
            onCancel();
            return;
        }
        const count = stage === 'provider' ? providers.length : models.length;
        const sel = stage === 'provider' ? providerIdx : modelIdx;
        const setSel = stage === 'provider' ? setProviderIdx : setModelIdx;
        if (key.upArrow && sel > 0) {
            setSel(v => v - 1);
            return;
        }
        if (key.downArrow && sel < count - 1) {
            setSel(v => v + 1);
            return;
        }
        if (key.return) {
            if (stage === 'provider') {
                if (!provider) {
                    return;
                }
                setStage('model');
                setModelIdx(0);
                return;
            }
            const model = models[modelIdx];
            if (provider && model) {
                onSelect(`${model} --provider ${provider.slug}${persistGlobal ? ' --global' : ''}`);
            }
            else {
                setStage('provider');
            }
            return;
        }
        if (ch.toLowerCase() === 'g') {
            setPersistGlobal(v => !v);
            return;
        }
        const n = ch === '0' ? 10 : parseInt(ch, 10);
        if (!Number.isNaN(n) && n >= 1 && n <= Math.min(10, count)) {
            const off = pageOffset(count, sel);
            if (stage === 'provider') {
                const next = off + n - 1;
                if (providers[next]) {
                    setProviderIdx(next);
                }
            }
            else if (provider && models[off + n - 1]) {
                onSelect(`${models[off + n - 1]} --provider ${provider.slug}${persistGlobal ? ' --global' : ''}`);
            }
        }
    });
    if (loading) {
        return _jsx(Text, { color: t.color.dim, children: "loading models\u2026" });
    }
    if (err) {
        return (_jsxs(Box, { flexDirection: "column", children: [_jsxs(Text, { color: t.color.label, children: ["error: ", err] }), _jsx(Text, { color: t.color.dim, children: "Esc to cancel" })] }));
    }
    if (!providers.length) {
        return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { color: t.color.dim, children: "no authenticated providers" }), _jsx(Text, { color: t.color.dim, children: "Esc to cancel" })] }));
    }
    if (stage === 'provider') {
        const rows = providers.map((p, i) => `${p.is_current ? '*' : ' '} ${names[i]} · ${p.total_models ?? p.models?.length ?? 0} models`);
        const { items, off } = visibleItems(rows, providerIdx);
        return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { bold: true, color: t.color.amber, children: "Select Provider" }), _jsxs(Text, { color: t.color.dim, children: ["Current model: ", currentModel || '(unknown)'] }), provider?.warning ? _jsxs(Text, { color: t.color.label, children: ["warning: ", provider.warning] }) : null, off > 0 && _jsxs(Text, { color: t.color.dim, children: [" \u2191 ", off, " more"] }), items.map((row, i) => {
                    const idx = off + i;
                    return (_jsxs(Text, { color: providerIdx === idx ? t.color.cornsilk : t.color.dim, children: [providerIdx === idx ? '▸ ' : '  ', i + 1, ". ", row] }, providers[idx]?.slug ?? `row-${idx}`));
                }), off + VISIBLE < rows.length && _jsxs(Text, { color: t.color.dim, children: [" \u2193 ", rows.length - off - VISIBLE, " more"] }), _jsxs(Text, { color: t.color.dim, children: ["persist: ", persistGlobal ? 'global' : 'session', " \u00B7 g toggle"] }), _jsx(Text, { color: t.color.dim, children: "\u2191/\u2193 select \u00B7 Enter choose \u00B7 1-9,0 quick \u00B7 Esc cancel" })] }));
    }
    const { items, off } = visibleItems(models, modelIdx);
    return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { bold: true, color: t.color.amber, children: "Select Model" }), _jsx(Text, { color: t.color.dim, children: names[providerIdx] || '(unknown provider)' }), !models.length ? _jsx(Text, { color: t.color.dim, children: "no models listed for this provider" }) : null, provider?.warning ? _jsxs(Text, { color: t.color.label, children: ["warning: ", provider.warning] }) : null, off > 0 && _jsxs(Text, { color: t.color.dim, children: [" \u2191 ", off, " more"] }), items.map((row, i) => {
                const idx = off + i;
                return (_jsxs(Text, { color: modelIdx === idx ? t.color.cornsilk : t.color.dim, children: [modelIdx === idx ? '▸ ' : '  ', i + 1, ". ", row] }, `${provider?.slug ?? 'prov'}:${idx}:${row}`));
            }), off + VISIBLE < models.length && _jsxs(Text, { color: t.color.dim, children: [" \u2193 ", models.length - off - VISIBLE, " more"] }), _jsxs(Text, { color: t.color.dim, children: ["persist: ", persistGlobal ? 'global' : 'session', " \u00B7 g toggle"] }), _jsx(Text, { color: t.color.dim, children: models.length ? '↑/↓ select · Enter switch · 1-9,0 quick · Esc back' : 'Enter/Esc back' })] }));
}
