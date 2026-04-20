import { createElement as _createElement } from "react";
import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { Box, NoSelect, Text } from '@hermes/ink';
import { memo, useEffect, useMemo, useState } from 'react';
import spinners from 'unicode-animations';
import { THINKING_COT_MAX } from '../config/limits.js';
import { compactPreview, estimateTokensRough, fmtK, formatToolCall, parseToolTrailResultLine, pick, thinkingPreview, toolTrailLabel } from '../lib/text.js';
const THINK = ['helix', 'breathe', 'orbit', 'dna', 'waverows', 'snake', 'pulse'];
const TOOL = ['cascade', 'scan', 'diagswipe', 'fillsweep', 'rain', 'columns', 'sparkle'];
const fmtElapsed = (ms) => {
    const sec = Math.max(0, ms) / 1000;
    return sec < 10 ? `${sec.toFixed(1)}s` : `${Math.round(sec)}s`;
};
const nextTreeRails = (rails, branch) => [...rails, branch === 'mid'];
const treeLead = (rails, branch) => `${rails.map(on => (on ? '│ ' : '  ')).join('')}${branch === 'mid' ? '├─ ' : '└─ '}`;
// ── Primitives ───────────────────────────────────────────────────────
function TreeRow({ branch, children, rails = [], stemColor, stemDim = true, t }) {
    const lead = treeLead(rails, branch);
    return (_jsxs(Box, { children: [_jsx(NoSelect, { flexShrink: 0, fromLeftEdge: true, width: lead.length, children: _jsx(Text, { color: stemColor ?? t.color.dim, dim: stemDim, children: lead }) }), _jsx(Box, { flexDirection: "column", flexGrow: 1, children: children })] }));
}
function TreeTextRow({ branch, color, content, dimColor, rails = [], t, wrap = 'wrap-trim' }) {
    const text = dimColor ? (_jsx(Text, { color: color, dim: true, wrap: wrap, children: content })) : (_jsx(Text, { color: color, wrap: wrap, children: content }));
    return (_jsx(TreeRow, { branch: branch, rails: rails, t: t, children: text }));
}
function TreeNode({ branch, children, header, open, rails = [], t }) {
    return (_jsxs(Box, { flexDirection: "column", children: [_jsx(TreeRow, { branch: branch, rails: rails, t: t, children: header }), open ? children?.(nextTreeRails(rails, branch)) : null] }));
}
export function Spinner({ color, variant = 'think' }) {
    const spin = useMemo(() => {
        const raw = spinners[pick(variant === 'tool' ? TOOL : THINK)];
        return { ...raw, frames: raw.frames.map(f => [...f][0] ?? '⠀') };
    }, [variant]);
    const [frame, setFrame] = useState(0);
    useEffect(() => {
        setFrame(0);
    }, [spin]);
    useEffect(() => {
        const id = setInterval(() => setFrame(f => (f + 1) % spin.frames.length), spin.interval);
        return () => clearInterval(id);
    }, [spin]);
    return _jsx(Text, { color: color, children: spin.frames[frame] });
}
function Detail({ branch = 'last', color, content, dimColor, rails = [], t }) {
    return _jsx(TreeTextRow, { branch: branch, color: color, content: content, dimColor: dimColor, rails: rails, t: t });
}
function StreamCursor({ color, dimColor, streaming = false, visible = false }) {
    const [on, setOn] = useState(true);
    useEffect(() => {
        if (!visible || !streaming) {
            setOn(true);
            return;
        }
        const id = setInterval(() => setOn(v => !v), 420);
        return () => clearInterval(id);
    }, [streaming, visible]);
    if (!visible) {
        return null;
    }
    return dimColor ? (_jsx(Text, { color: color, dim: true, children: streaming && on ? '▍' : ' ' })) : (_jsx(Text, { color: color, children: streaming && on ? '▍' : ' ' }));
}
function Chevron({ count, onClick, open, suffix, t, title, tone = 'dim' }) {
    const color = tone === 'error' ? t.color.error : tone === 'warn' ? t.color.warn : t.color.dim;
    return (_jsx(Box, { onClick: (e) => onClick(!!e?.shiftKey || !!e?.ctrlKey), children: _jsxs(Text, { color: color, dim: tone === 'dim', children: [_jsx(Text, { color: t.color.amber, children: open ? '▾ ' : '▸ ' }), title, typeof count === 'number' ? ` (${count})` : '', suffix ? (_jsxs(Text, { color: t.color.statusFg, dim: true, children: ['  ', suffix] })) : null] }) }));
}
function SubagentAccordion({ branch, expanded, item, rails = [], t }) {
    const [open, setOpen] = useState(expanded);
    const [deep, setDeep] = useState(expanded);
    const [openThinking, setOpenThinking] = useState(expanded);
    const [openTools, setOpenTools] = useState(expanded);
    const [openNotes, setOpenNotes] = useState(expanded);
    useEffect(() => {
        if (!expanded) {
            return;
        }
        setOpen(true);
        setDeep(true);
        setOpenThinking(true);
        setOpenTools(true);
        setOpenNotes(true);
    }, [expanded]);
    const expandAll = () => {
        setOpen(true);
        setDeep(true);
        setOpenThinking(true);
        setOpenTools(true);
        setOpenNotes(true);
    };
    const statusTone = item.status === 'failed' ? 'error' : item.status === 'interrupted' ? 'warn' : 'dim';
    const prefix = item.taskCount > 1 ? `[${item.index + 1}/${item.taskCount}] ` : '';
    const goalLabel = item.goal || `Subagent ${item.index + 1}`;
    const title = `${prefix}${open ? goalLabel : compactPreview(goalLabel, 60)}`;
    const summary = compactPreview((item.summary || '').replace(/\s+/g, ' ').trim(), 72);
    const suffix = item.status === 'running'
        ? 'running'
        : `${item.status}${item.durationSeconds ? ` · ${fmtElapsed(item.durationSeconds * 1000)}` : ''}`;
    const thinkingText = item.thinking.join('\n');
    const hasThinking = Boolean(thinkingText);
    const hasTools = item.tools.length > 0;
    const noteRows = [...(summary ? [summary] : []), ...item.notes];
    const hasNotes = noteRows.length > 0;
    const showChildren = expanded || deep;
    const noteColor = statusTone === 'error' ? t.color.error : statusTone === 'warn' ? t.color.warn : t.color.dim;
    const sections = [];
    if (hasThinking) {
        sections.push({
            header: (_jsx(Chevron, { count: item.thinking.length, onClick: shift => {
                    if (shift) {
                        expandAll();
                    }
                    else {
                        setOpenThinking(v => !v);
                    }
                }, open: showChildren || openThinking, t: t, title: "Thinking" })),
            key: 'thinking',
            open: showChildren || openThinking,
            render: childRails => (_jsx(Thinking, { active: item.status === 'running', branch: "last", mode: "full", rails: childRails, reasoning: thinkingText, streaming: item.status === 'running', t: t }))
        });
    }
    if (hasTools) {
        sections.push({
            header: (_jsx(Chevron, { count: item.tools.length, onClick: shift => {
                    if (shift) {
                        expandAll();
                    }
                    else {
                        setOpenTools(v => !v);
                    }
                }, open: showChildren || openTools, t: t, title: "Tool calls" })),
            key: 'tools',
            open: showChildren || openTools,
            render: childRails => (_jsx(Box, { flexDirection: "column", children: item.tools.map((line, index) => (_jsx(TreeTextRow, { branch: index === item.tools.length - 1 ? 'last' : 'mid', color: t.color.cornsilk, content: _jsxs(_Fragment, { children: [_jsx(Text, { color: t.color.amber, children: "\u25CF " }), line] }), rails: childRails, t: t }, `${item.id}-tool-${index}`))) }))
        });
    }
    if (hasNotes) {
        sections.push({
            header: (_jsx(Chevron, { count: noteRows.length, onClick: shift => {
                    if (shift) {
                        expandAll();
                    }
                    else {
                        setOpenNotes(v => !v);
                    }
                }, open: showChildren || openNotes, t: t, title: "Progress", tone: statusTone })),
            key: 'notes',
            open: showChildren || openNotes,
            render: childRails => (_jsx(Box, { flexDirection: "column", children: noteRows.map((line, index) => (_jsx(TreeTextRow, { branch: index === noteRows.length - 1 ? 'last' : 'mid', color: noteColor, content: line, dimColor: statusTone === 'dim', rails: childRails, t: t }, `${item.id}-note-${index}`))) }))
        });
    }
    return (_jsx(TreeNode, { branch: branch, header: _jsx(Chevron, { onClick: shift => {
                if (shift) {
                    expandAll();
                    return;
                }
                setOpen(v => {
                    if (!v) {
                        setDeep(false);
                    }
                    return !v;
                });
            }, open: open, suffix: suffix, t: t, title: title, tone: statusTone }), open: open, rails: rails, t: t, children: childRails => (_jsx(Box, { flexDirection: "column", children: sections.map((section, index) => (_jsx(TreeNode, { branch: index === sections.length - 1 ? 'last' : 'mid', header: section.header, open: section.open, rails: childRails, t: t, children: section.render }, `${item.id}-${section.key}`))) })) }));
}
// ── Thinking ─────────────────────────────────────────────────────────
export const Thinking = memo(function Thinking({ active = false, branch = 'last', mode = 'truncated', rails = [], reasoning, streaming = false, t }) {
    const preview = useMemo(() => thinkingPreview(reasoning, mode, THINKING_COT_MAX), [mode, reasoning]);
    const lines = useMemo(() => preview.split('\n').map(line => line.replace(/\t/g, '  ')), [preview]);
    if (!preview && !active) {
        return null;
    }
    return (_jsx(TreeRow, { branch: branch, rails: rails, t: t, children: _jsx(Box, { flexDirection: "column", flexGrow: 1, children: preview ? (mode === 'full' ? (lines.map((line, index) => (_jsxs(Text, { color: t.color.dim, dim: true, wrap: "wrap-trim", children: [line || ' ', index === lines.length - 1 ? (_jsx(StreamCursor, { color: t.color.dim, dimColor: true, streaming: streaming, visible: active })) : null] }, index)))) : (_jsxs(Text, { color: t.color.dim, dim: true, wrap: "truncate-end", children: [preview, _jsx(StreamCursor, { color: t.color.dim, dimColor: true, streaming: streaming, visible: active })] }))) : (_jsx(Text, { color: t.color.dim, dim: true, children: _jsx(StreamCursor, { color: t.color.dim, dimColor: true, streaming: streaming, visible: active }) })) }) }));
});
export const ToolTrail = memo(function ToolTrail({ busy = false, detailsMode = 'collapsed', outcome = '', reasoningActive = false, reasoning = '', reasoningTokens, reasoningStreaming = false, subagents = [], t, tools = [], toolTokens, trail = [], activity = [] }) {
    const [now, setNow] = useState(() => Date.now());
    const [openThinking, setOpenThinking] = useState(false);
    const [openTools, setOpenTools] = useState(false);
    const [openSubagents, setOpenSubagents] = useState(false);
    const [deepSubagents, setDeepSubagents] = useState(false);
    const [openMeta, setOpenMeta] = useState(false);
    useEffect(() => {
        if (!tools.length || (detailsMode === 'collapsed' && !openTools)) {
            return;
        }
        const id = setInterval(() => setNow(Date.now()), 500);
        return () => clearInterval(id);
    }, [detailsMode, openTools, tools.length]);
    useEffect(() => {
        if (detailsMode === 'expanded') {
            setOpenThinking(true);
            setOpenTools(true);
            setOpenSubagents(true);
            setOpenMeta(true);
        }
        if (detailsMode === 'hidden') {
            setOpenThinking(false);
            setOpenTools(false);
            setOpenSubagents(false);
            setOpenMeta(false);
        }
    }, [detailsMode]);
    const cot = useMemo(() => thinkingPreview(reasoning, 'full', THINKING_COT_MAX), [reasoning]);
    if (!busy &&
        !trail.length &&
        !tools.length &&
        !subagents.length &&
        !activity.length &&
        !cot &&
        !reasoningActive &&
        !outcome) {
        return null;
    }
    // ── Build groups + meta ────────────────────────────────────────
    const groups = [];
    const meta = [];
    const pushDetail = (row) => (groups.at(-1)?.details ?? meta).push(row);
    for (const [i, line] of trail.entries()) {
        const parsed = parseToolTrailResultLine(line);
        if (parsed) {
            groups.push({
                color: parsed.mark === '✗' ? t.color.error : t.color.cornsilk,
                content: parsed.detail ? parsed.call : `${parsed.call} ${parsed.mark}`,
                details: [],
                key: `tr-${i}`,
                label: parsed.call
            });
            if (parsed.detail) {
                pushDetail({
                    color: parsed.mark === '✗' ? t.color.error : t.color.dim,
                    content: parsed.detail,
                    dimColor: parsed.mark !== '✗',
                    key: `tr-${i}-d`
                });
            }
            continue;
        }
        if (line.startsWith('drafting ')) {
            const label = toolTrailLabel(line.slice(9).replace(/…$/, '').trim());
            groups.push({
                color: t.color.cornsilk,
                content: label,
                details: [{ color: t.color.dim, content: 'drafting...', dimColor: true, key: `tr-${i}-d` }],
                key: `tr-${i}`,
                label
            });
            continue;
        }
        if (line === 'analyzing tool output…') {
            pushDetail({
                color: t.color.dim,
                dimColor: true,
                key: `tr-${i}`,
                content: groups.length ? (_jsxs(_Fragment, { children: [_jsx(Spinner, { color: t.color.amber, variant: "think" }), " ", line] })) : (line)
            });
            continue;
        }
        meta.push({ color: t.color.dim, content: line, dimColor: true, key: `tr-${i}` });
    }
    for (const tool of tools) {
        const label = formatToolCall(tool.name, tool.context || '');
        groups.push({
            color: t.color.cornsilk,
            key: tool.id,
            label,
            details: [],
            content: (_jsxs(_Fragment, { children: [_jsx(Spinner, { color: t.color.amber, variant: "tool" }), " ", label, tool.startedAt ? ` (${fmtElapsed(now - tool.startedAt)})` : ''] }))
        });
    }
    for (const item of activity.slice(-4)) {
        const glyph = item.tone === 'error' ? '✗' : item.tone === 'warn' ? '!' : '·';
        const color = item.tone === 'error' ? t.color.error : item.tone === 'warn' ? t.color.warn : t.color.dim;
        meta.push({ color, content: `${glyph} ${item.text}`, dimColor: item.tone === 'info', key: `a-${item.id}` });
    }
    // ── Derived ────────────────────────────────────────────────────
    const hasTools = groups.length > 0;
    const hasSubagents = subagents.length > 0;
    const hasMeta = meta.length > 0;
    const hasThinking = !!cot || reasoningActive || busy;
    const thinkingLive = reasoningActive || reasoningStreaming;
    const tokenCount = reasoningTokens && reasoningTokens > 0 ? reasoningTokens : reasoning ? estimateTokensRough(reasoning) : 0;
    const toolTokenCount = toolTokens ?? 0;
    const totalTokenCount = tokenCount + toolTokenCount;
    const thinkingTokensLabel = tokenCount > 0 ? `~${fmtK(tokenCount)} tokens` : null;
    const toolTokensLabel = toolTokens !== undefined && toolTokens > 0 ? `~${fmtK(toolTokens)} tokens` : undefined;
    const totalTokensLabel = tokenCount > 0 && toolTokenCount > 0 ? `~${fmtK(totalTokenCount)} total` : null;
    const delegateGroups = groups.filter(g => g.label.startsWith('Delegate Task'));
    const inlineDelegateKey = hasSubagents && delegateGroups.length === 1 ? delegateGroups[0].key : null;
    // ── Hidden: errors/warnings only ──────────────────────────────
    if (detailsMode === 'hidden') {
        const alerts = activity.filter(i => i.tone !== 'info').slice(-2);
        return alerts.length ? (_jsx(Box, { flexDirection: "column", children: alerts.map(i => (_jsxs(Text, { color: i.tone === 'error' ? t.color.error : t.color.warn, children: [i.tone === 'error' ? '✗' : '!', " ", i.text] }, `ha-${i.id}`))) })) : null;
    }
    // ── Tree render fragments ──────────────────────────────────────
    const expandAll = () => {
        setOpenThinking(true);
        setOpenTools(true);
        setOpenSubagents(true);
        setDeepSubagents(true);
        setOpenMeta(true);
    };
    const metaTone = activity.some(i => i.tone === 'error')
        ? 'error'
        : activity.some(i => i.tone === 'warn')
            ? 'warn'
            : 'dim';
    const renderSubagentList = (rails) => (_jsx(Box, { flexDirection: "column", children: subagents.map((item, index) => (_jsx(SubagentAccordion, { branch: index === subagents.length - 1 ? 'last' : 'mid', expanded: detailsMode === 'expanded' || deepSubagents, item: item, rails: rails, t: t }, item.id))) }));
    const sections = [];
    if (hasThinking) {
        sections.push({
            header: (_jsx(Box, { onClick: (e) => {
                    if (e?.shiftKey || e?.ctrlKey) {
                        expandAll();
                    }
                    else {
                        setOpenThinking(v => !v);
                    }
                }, children: _jsxs(Text, { color: t.color.dim, dim: !thinkingLive, children: [_jsx(Text, { color: t.color.amber, children: detailsMode === 'expanded' || openThinking ? '▾ ' : '▸ ' }), thinkingLive ? (_jsx(Text, { bold: true, color: t.color.cornsilk, children: "Thinking" })) : (_jsx(Text, { color: t.color.dim, dim: true, children: "Thinking" })), thinkingTokensLabel ? (_jsxs(Text, { color: t.color.statusFg, dim: true, children: ['  ', thinkingTokensLabel] })) : null] }) })),
            key: 'thinking',
            open: detailsMode === 'expanded' || openThinking,
            render: rails => (_jsx(Thinking, { active: reasoningActive, branch: "last", mode: "full", rails: rails, reasoning: busy ? reasoning : cot, streaming: busy && reasoningStreaming, t: t }))
        });
    }
    if (hasTools) {
        sections.push({
            header: (_jsx(Chevron, { count: groups.length, onClick: shift => {
                    if (shift) {
                        expandAll();
                    }
                    else {
                        setOpenTools(v => !v);
                    }
                }, open: detailsMode === 'expanded' || openTools, suffix: toolTokensLabel, t: t, title: "Tool calls" })),
            key: 'tools',
            open: detailsMode === 'expanded' || openTools,
            render: rails => (_jsx(Box, { flexDirection: "column", children: groups.map((group, index) => {
                    const branch = index === groups.length - 1 ? 'last' : 'mid';
                    const childRails = nextTreeRails(rails, branch);
                    const hasInlineSubagents = inlineDelegateKey === group.key;
                    return (_jsxs(Box, { flexDirection: "column", children: [_jsx(TreeTextRow, { branch: branch, color: group.color, content: _jsxs(_Fragment, { children: [_jsx(Text, { color: t.color.amber, children: "\u25CF " }), group.content] }), rails: rails, t: t }), group.details.map((detail, detailIndex) => (_createElement(Detail, { ...detail, branch: detailIndex === group.details.length - 1 && !hasInlineSubagents ? 'last' : 'mid', key: detail.key, rails: childRails, t: t }))), hasInlineSubagents ? renderSubagentList(childRails) : null] }, group.key));
                }) }))
        });
    }
    if (hasSubagents && !inlineDelegateKey) {
        sections.push({
            header: (_jsx(Chevron, { count: subagents.length, onClick: shift => {
                    if (shift) {
                        expandAll();
                        setDeepSubagents(true);
                    }
                    else {
                        setOpenSubagents(v => !v);
                        setDeepSubagents(false);
                    }
                }, open: detailsMode === 'expanded' || openSubagents, t: t, title: "Subagents" })),
            key: 'subagents',
            open: detailsMode === 'expanded' || openSubagents,
            render: renderSubagentList
        });
    }
    if (hasMeta) {
        sections.push({
            header: (_jsx(Chevron, { count: meta.length, onClick: shift => {
                    if (shift) {
                        expandAll();
                    }
                    else {
                        setOpenMeta(v => !v);
                    }
                }, open: detailsMode === 'expanded' || openMeta, t: t, title: "Activity", tone: metaTone })),
            key: 'meta',
            open: detailsMode === 'expanded' || openMeta,
            render: rails => (_jsx(Box, { flexDirection: "column", children: meta.map((row, index) => (_jsx(TreeTextRow, { branch: index === meta.length - 1 ? 'last' : 'mid', color: row.color, content: row.content, dimColor: row.dimColor, rails: rails, t: t }, row.key))) }))
        });
    }
    const topCount = sections.length + (totalTokensLabel ? 1 : 0);
    return (_jsxs(Box, { flexDirection: "column", children: [sections.map((section, index) => (_jsx(TreeNode, { branch: index === topCount - 1 ? 'last' : 'mid', header: section.header, open: section.open, t: t, children: section.render }, section.key))), totalTokensLabel ? (_jsx(TreeTextRow, { branch: "last", color: t.color.statusFg, content: _jsxs(_Fragment, { children: [_jsx(Text, { color: t.color.amber, children: "\u03A3 " }), totalTokensLabel] }), dimColor: true, t: t })) : null, outcome ? (_jsx(Box, { marginTop: 1, children: _jsxs(Text, { color: t.color.dim, dim: true, children: ["\u00B7 ", outcome] }) })) : null] }));
});
