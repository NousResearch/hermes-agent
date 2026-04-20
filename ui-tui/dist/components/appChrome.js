import { jsxs as _jsxs, jsx as _jsx, Fragment as _Fragment } from "react/jsx-runtime";
import { Box, Text } from '@hermes/ink';
import { useCallback, useEffect, useState, useSyncExternalStore } from 'react';
import { FACES } from '../content/faces.js';
import { VERBS } from '../content/verbs.js';
import { fmtDuration } from '../domain/messages.js';
import { stickyPromptFromViewport } from '../domain/viewport.js';
import { fmtK } from '../lib/text.js';
const FACE_TICK_MS = 2500;
const HEART_COLORS = ['#ff5fa2', '#ff4d6d'];
function FaceTicker({ color }) {
    const [tick, setTick] = useState(() => Math.floor(Math.random() * 1000));
    useEffect(() => {
        const id = setInterval(() => setTick(n => n + 1), FACE_TICK_MS);
        return () => clearInterval(id);
    }, []);
    return (_jsxs(Text, { color: color, children: [FACES[tick % FACES.length], " ", VERBS[tick % VERBS.length], "\u2026"] }));
}
function ctxBarColor(pct, t) {
    if (pct == null) {
        return t.color.dim;
    }
    if (pct >= 95) {
        return t.color.statusCritical;
    }
    if (pct > 80) {
        return t.color.statusBad;
    }
    if (pct >= 50) {
        return t.color.statusWarn;
    }
    return t.color.statusGood;
}
function ctxBar(pct, w = 10) {
    const p = Math.max(0, Math.min(100, pct ?? 0));
    const filled = Math.round((p / 100) * w);
    return '█'.repeat(filled) + '░'.repeat(w - filled);
}
function SessionDuration({ startedAt }) {
    const [now, setNow] = useState(() => Date.now());
    useEffect(() => {
        setNow(Date.now());
        const id = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(id);
    }, [startedAt]);
    return fmtDuration(now - startedAt);
}
export function GoodVibesHeart({ tick, t }) {
    const [active, setActive] = useState(false);
    const [color, setColor] = useState(t.color.amber);
    useEffect(() => {
        if (tick <= 0) {
            return;
        }
        const palette = [...HEART_COLORS, t.color.amber];
        setColor(palette[Math.floor(Math.random() * palette.length)]);
        setActive(true);
        const id = setTimeout(() => setActive(false), 650);
        return () => clearTimeout(id);
    }, [t.color.amber, tick]);
    return _jsx(Text, { color: color, children: active ? '♥' : ' ' });
}
export function StatusRule({ cwdLabel, cols, busy, status, statusColor, model, usage, bgCount, sessionStartedAt, showCost, voiceLabel, t }) {
    const pct = usage.context_percent;
    const barColor = ctxBarColor(pct, t);
    const ctxLabel = usage.context_max
        ? `${fmtK(usage.context_used ?? 0)}/${fmtK(usage.context_max)}`
        : usage.total > 0
            ? `${fmtK(usage.total)} tok`
            : '';
    const bar = usage.context_max ? ctxBar(pct) : '';
    const leftWidth = Math.max(12, cols - cwdLabel.length - 3);
    return (_jsxs(Box, { children: [_jsx(Box, { flexShrink: 1, width: leftWidth, children: _jsxs(Text, { color: t.color.bronze, wrap: "truncate-end", children: ['─ ', busy ? _jsx(FaceTicker, { color: statusColor }) : _jsx(Text, { color: statusColor, children: status }), _jsxs(Text, { color: t.color.dim, children: [" \u2502 ", model] }), ctxLabel ? _jsxs(Text, { color: t.color.dim, children: [" \u2502 ", ctxLabel] }) : null, bar ? (_jsxs(Text, { color: t.color.dim, children: [' │ ', _jsxs(Text, { color: barColor, children: ["[", bar, "]"] }), " ", _jsx(Text, { color: barColor, children: pct != null ? `${pct}%` : '' })] })) : null, sessionStartedAt ? (_jsxs(Text, { color: t.color.dim, children: [' │ ', _jsx(SessionDuration, { startedAt: sessionStartedAt })] })) : null, voiceLabel ? _jsxs(Text, { color: t.color.dim, children: [" \u2502 ", voiceLabel] }) : null, bgCount > 0 ? _jsxs(Text, { color: t.color.dim, children: [" \u2502 ", bgCount, " bg"] }) : null, showCost && typeof usage.cost_usd === 'number' ? (_jsxs(Text, { color: t.color.dim, children: [" \u2502 $", usage.cost_usd.toFixed(4)] })) : null] }) }), _jsx(Text, { color: t.color.bronze, children: " \u2500 " }), _jsx(Text, { color: t.color.label, children: cwdLabel })] }));
}
export function FloatBox({ children, color }) {
    return (_jsx(Box, { alignSelf: "flex-start", borderColor: color, borderStyle: "double", flexDirection: "column", marginTop: 1, opaque: true, paddingX: 1, children: children }));
}
export function StickyPromptTracker({ messages, offsets, scrollRef, onChange }) {
    useSyncExternalStore(useCallback((cb) => scrollRef.current?.subscribe(cb) ?? (() => { }), [scrollRef]), () => {
        const s = scrollRef.current;
        if (!s) {
            return NaN;
        }
        const top = Math.max(0, s.getScrollTop() + s.getPendingDelta());
        return s.isSticky() ? -1 - top : top;
    }, () => NaN);
    const s = scrollRef.current;
    const top = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0));
    const text = stickyPromptFromViewport(messages, offsets, top, s?.isSticky() ?? true);
    useEffect(() => onChange(text), [onChange, text]);
    return null;
}
export function TranscriptScrollbar({ scrollRef, t }) {
    useSyncExternalStore(useCallback((cb) => scrollRef.current?.subscribe(cb) ?? (() => { }), [scrollRef]), () => {
        const s = scrollRef.current;
        if (!s) {
            return NaN;
        }
        const vp = Math.max(0, s.getViewportHeight());
        const total = Math.max(vp, s.getScrollHeight());
        const top = Math.max(0, s.getScrollTop() + s.getPendingDelta());
        const thumb = total > vp ? Math.max(1, Math.round((vp * vp) / total)) : vp;
        const travel = Math.max(1, vp - thumb);
        const thumbTop = total > vp ? Math.round((top / Math.max(1, total - vp)) * travel) : 0;
        return `${thumbTop}:${thumb}:${vp}`;
    }, () => '');
    const [hover, setHover] = useState(false);
    const [grab, setGrab] = useState(null);
    const s = scrollRef.current;
    const vp = Math.max(0, s?.getViewportHeight() ?? 0);
    if (!vp) {
        return _jsx(Box, { width: 1 });
    }
    const total = Math.max(vp, s?.getScrollHeight() ?? vp);
    const scrollable = total > vp;
    const thumb = scrollable ? Math.max(1, Math.round((vp * vp) / total)) : vp;
    const travel = Math.max(1, vp - thumb);
    const pos = Math.max(0, (s?.getScrollTop() ?? 0) + (s?.getPendingDelta() ?? 0));
    const thumbTop = scrollable ? Math.round((pos / Math.max(1, total - vp)) * travel) : 0;
    const thumbColor = grab !== null ? t.color.gold : hover ? t.color.amber : t.color.bronze;
    const trackColor = hover ? t.color.bronze : t.color.dim;
    const jump = (row, offset) => {
        if (!s || !scrollable) {
            return;
        }
        s.scrollTo(Math.round((Math.max(0, Math.min(travel, row - offset)) / travel) * Math.max(0, total - vp)));
    };
    return (_jsx(Box, { flexDirection: "column", onMouseDown: (e) => {
            const row = Math.max(0, Math.min(vp - 1, e.localRow ?? 0));
            const off = row >= thumbTop && row < thumbTop + thumb ? row - thumbTop : Math.floor(thumb / 2);
            setGrab(off);
            jump(row, off);
        }, onMouseDrag: (e) => jump(Math.max(0, Math.min(vp - 1, e.localRow ?? 0)), grab ?? Math.floor(thumb / 2)), onMouseEnter: () => setHover(true), onMouseLeave: () => setHover(false), onMouseUp: () => setGrab(null), width: 1, children: !scrollable ? (_jsxs(Text, { color: trackColor, dim: true, children: [' \n'.repeat(Math.max(0, vp - 1)), ' '] })) : (_jsxs(_Fragment, { children: [thumbTop > 0 ? (_jsx(Text, { color: trackColor, dim: !hover, children: `${'│\n'.repeat(Math.max(0, thumbTop - 1))}${thumbTop > 0 ? '│' : ''}` })) : null, thumb > 0 ? (_jsx(Text, { color: thumbColor, children: `${'┃\n'.repeat(Math.max(0, thumb - 1))}${thumb > 0 ? '┃' : ''}` })) : null, vp - thumbTop - thumb > 0 ? (_jsx(Text, { color: trackColor, dim: !hover, children: `${'│\n'.repeat(Math.max(0, vp - thumbTop - thumb - 1))}${vp - thumbTop - thumb > 0 ? '│' : ''}` })) : null] })) }));
}
