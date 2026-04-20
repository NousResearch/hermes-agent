import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { Box, Text, useInput } from '@hermes/ink';
import { useState } from 'react';
import { TextInput } from './textInput.js';
const OPTS = ['once', 'session', 'always', 'deny'];
const LABELS = { always: 'Always allow', deny: 'Deny', once: 'Allow once', session: 'Allow this session' };
const CMD_PREVIEW_LINES = 10;
export function ApprovalPrompt({ onChoice, req, t }) {
    const [sel, setSel] = useState(0);
    useInput((ch, key) => {
        if (key.upArrow && sel > 0) {
            setSel(s => s - 1);
        }
        if (key.downArrow && sel < OPTS.length - 1) {
            setSel(s => s + 1);
        }
        const n = parseInt(ch, 10);
        if (n >= 1 && n <= OPTS.length) {
            onChoice(OPTS[n - 1]);
            return;
        }
        if (key.return) {
            onChoice(OPTS[sel]);
        }
    });
    const rawLines = req.command.split('\n');
    const shown = rawLines.slice(0, CMD_PREVIEW_LINES);
    const overflow = rawLines.length - shown.length;
    return (_jsxs(Box, { borderColor: t.color.warn, borderStyle: "double", flexDirection: "column", paddingX: 1, children: [_jsxs(Text, { bold: true, color: t.color.warn, children: ["\u26A0 approval required \u00B7 ", req.description] }), _jsxs(Box, { flexDirection: "column", paddingLeft: 1, children: [shown.map((line, i) => (_jsx(Text, { color: t.color.cornsilk, wrap: "truncate-end", children: line || ' ' }, i))), overflow > 0 ? (_jsxs(Text, { color: t.color.dim, children: ["\u2026 +", overflow, " more line", overflow === 1 ? '' : 's', " (full text above)"] })) : null] }), _jsx(Text, {}), OPTS.map((o, i) => (_jsxs(Text, { children: [_jsx(Text, { color: sel === i ? t.color.warn : t.color.dim, children: sel === i ? '▸ ' : '  ' }), _jsxs(Text, { color: sel === i ? t.color.cornsilk : t.color.dim, children: [i + 1, ". ", LABELS[o]] })] }, o))), _jsx(Text, { color: t.color.dim, children: "\u2191/\u2193 select \u00B7 Enter confirm \u00B7 1-4 quick pick \u00B7 Ctrl+C deny" })] }));
}
export function ClarifyPrompt({ cols = 80, onAnswer, onCancel, req, t }) {
    const [sel, setSel] = useState(0);
    const [custom, setCustom] = useState('');
    const [typing, setTyping] = useState(false);
    const choices = req.choices ?? [];
    const heading = (_jsxs(Text, { bold: true, children: [_jsx(Text, { color: t.color.amber, children: "ask" }), _jsxs(Text, { color: t.color.cornsilk, children: [" ", req.question] })] }));
    useInput((ch, key) => {
        if (key.escape) {
            typing && choices.length ? setTyping(false) : onCancel();
            return;
        }
        if (typing || !choices.length) {
            return;
        }
        if (key.upArrow && sel > 0) {
            setSel(s => s - 1);
        }
        if (key.downArrow && sel < choices.length) {
            setSel(s => s + 1);
        }
        if (key.return) {
            sel === choices.length ? setTyping(true) : choices[sel] && onAnswer(choices[sel]);
        }
        const n = parseInt(ch);
        if (n >= 1 && n <= choices.length) {
            onAnswer(choices[n - 1]);
        }
    });
    if (typing || !choices.length) {
        return (_jsxs(Box, { flexDirection: "column", children: [heading, _jsxs(Box, { children: [_jsx(Text, { color: t.color.label, children: '> ' }), _jsx(TextInput, { columns: Math.max(20, cols - 6), onChange: setCustom, onSubmit: onAnswer, value: custom })] }), _jsxs(Text, { color: t.color.dim, children: ["Enter send \u00B7 Esc ", choices.length ? 'back' : 'cancel', " \u00B7 Ctrl+C cancel"] })] }));
    }
    return (_jsxs(Box, { flexDirection: "column", children: [heading, [...choices, 'Other (type your answer)'].map((c, i) => (_jsxs(Text, { children: [_jsx(Text, { color: sel === i ? t.color.label : t.color.dim, children: sel === i ? '▸ ' : '  ' }), _jsxs(Text, { color: sel === i ? t.color.cornsilk : t.color.dim, children: [i + 1, ". ", c] })] }, i))), _jsxs(Text, { color: t.color.dim, children: ["\u2191/\u2193 select \u00B7 Enter confirm \u00B7 1-", choices.length, " quick pick \u00B7 Esc/Ctrl+C cancel"] })] }));
}
export function ConfirmPrompt({ onCancel, onConfirm, req, t }) {
    const [sel, setSel] = useState(0);
    useInput((ch, key) => {
        const lower = ch.toLowerCase();
        if (key.escape || (key.ctrl && lower === 'c') || lower === 'n') {
            return onCancel();
        }
        if (lower === 'y') {
            return onConfirm();
        }
        if (key.upArrow) {
            setSel(0);
        }
        if (key.downArrow) {
            setSel(1);
        }
        if (key.return) {
            sel === 0 ? onCancel() : onConfirm();
        }
    });
    const accent = req.danger ? t.color.error : t.color.warn;
    const rows = [
        { color: t.color.cornsilk, label: req.cancelLabel ?? 'No' },
        { color: req.danger ? t.color.error : t.color.cornsilk, label: req.confirmLabel ?? 'Yes' }
    ];
    return (_jsxs(Box, { borderColor: accent, borderStyle: "double", flexDirection: "column", paddingX: 1, children: [_jsxs(Text, { bold: true, color: accent, children: [req.danger ? '⚠' : '?', " ", req.title] }), req.detail ? (_jsx(Box, { paddingLeft: 1, children: _jsx(Text, { color: t.color.cornsilk, wrap: "truncate-end", children: req.detail }) })) : null, _jsx(Text, {}), rows.map((row, i) => (_jsxs(Text, { children: [_jsx(Text, { color: sel === i ? accent : t.color.dim, children: sel === i ? '▸ ' : '  ' }), _jsx(Text, { color: sel === i ? row.color : t.color.dim, children: row.label })] }, row.label))), _jsx(Text, { color: t.color.dim, children: "\u2191/\u2193 select \u00B7 Enter confirm \u00B7 Y/N quick \u00B7 Esc cancel" })] }));
}
