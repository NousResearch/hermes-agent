import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Text, useInput } from '@hermes/ink';
import { useEffect, useState } from 'react';
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js';
const VISIBLE = 15;
const age = (ts) => {
    const d = (Date.now() / 1000 - ts) / 86400;
    if (d < 1) {
        return 'today';
    }
    if (d < 2) {
        return 'yesterday';
    }
    return `${Math.floor(d)}d ago`;
};
export function SessionPicker({ gw, onCancel, onSelect, t }) {
    const [items, setItems] = useState([]);
    const [err, setErr] = useState('');
    const [sel, setSel] = useState(0);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        gw.request('session.list', { limit: 20 })
            .then(raw => {
            const r = asRpcResult(raw);
            if (!r) {
                setErr('invalid response: session.list');
                setLoading(false);
                return;
            }
            setItems(r.sessions ?? []);
            setErr('');
            setLoading(false);
        })
            .catch((e) => {
            setErr(rpcErrorMessage(e));
            setLoading(false);
        });
    }, [gw]);
    useInput((ch, key) => {
        if (key.escape) {
            return onCancel();
        }
        if (key.upArrow && sel > 0) {
            setSel(s => s - 1);
        }
        if (key.downArrow && sel < items.length - 1) {
            setSel(s => s + 1);
        }
        if (key.return && items[sel]) {
            onSelect(items[sel].id);
        }
        const n = parseInt(ch);
        if (n >= 1 && n <= Math.min(9, items.length)) {
            onSelect(items[n - 1].id);
        }
    });
    if (loading) {
        return _jsx(Text, { color: t.color.dim, children: "loading sessions\u2026" });
    }
    if (err) {
        return (_jsxs(Box, { flexDirection: "column", children: [_jsxs(Text, { color: t.color.label, children: ["error: ", err] }), _jsx(Text, { color: t.color.dim, children: "Esc to cancel" })] }));
    }
    if (!items.length) {
        return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { color: t.color.dim, children: "no previous sessions" }), _jsx(Text, { color: t.color.dim, children: "Esc to cancel" })] }));
    }
    const off = Math.max(0, Math.min(sel - Math.floor(VISIBLE / 2), items.length - VISIBLE));
    return (_jsxs(Box, { flexDirection: "column", children: [_jsx(Text, { bold: true, color: t.color.amber, children: "Resume Session" }), off > 0 && _jsxs(Text, { color: t.color.dim, children: [" \u2191 ", off, " more"] }), items.slice(off, off + VISIBLE).map((s, vi) => {
                const i = off + vi;
                return (_jsxs(Box, { children: [_jsx(Text, { color: sel === i ? t.color.label : t.color.dim, children: sel === i ? '▸ ' : '  ' }), _jsx(Box, { width: 30, children: _jsxs(Text, { color: sel === i ? t.color.cornsilk : t.color.dim, children: [String(i + 1).padStart(2), ". [", s.id, "]"] }) }), _jsx(Box, { width: 30, children: _jsxs(Text, { color: t.color.dim, children: ["(", s.message_count, " msgs, ", age(s.started_at), ", ", s.source || 'tui', ")"] }) }), _jsx(Text, { color: sel === i ? t.color.cornsilk : t.color.dim, children: s.title || s.preview || '(untitled)' })] }, s.id));
            }), off + VISIBLE < items.length && _jsxs(Text, { color: t.color.dim, children: [" \u2193 ", items.length - off - VISIBLE, " more"] }), _jsx(Text, { color: t.color.dim, children: "\u2191/\u2193 select \u00B7 Enter resume \u00B7 1-9 quick \u00B7 Esc cancel" })] }));
}
