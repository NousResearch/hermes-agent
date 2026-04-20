import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Link, Text } from '@hermes/ink';
import { memo, useMemo } from 'react';
import { highlightLine, isHighlightable } from '../lib/syntax.js';
const FENCE_RE = /^\s*(`{3,}|~{3,})(.*)$/;
const HR_RE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+\s*)?$/;
const FOOTNOTE_RE = /^\[\^([^\]]+)\]:\s*(.*)$/;
const DEF_RE = /^\s*:\s+(.+)$/;
const TABLE_DIVIDER_CELL_RE = /^:?-{3,}:?$/;
const MD_URL_RE = '((?:[^\\s()]|\\([^\\s()]*\\))+?)';
const INLINE_RE = new RegExp(`(!\\[(.*?)\\]\\(${MD_URL_RE}\\)|\\[(.+?)\\]\\(${MD_URL_RE}\\)|<((?:https?:\\/\\/|mailto:)[^>\\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,})>|~~(.+?)~~|\`([^\\\`]+)\`|\\*\\*(.+?)\\*\\*|__(.+?)__|\\*(.+?)\\*|_(.+?)_|==(.+?)==|\\[\\^([^\\]]+)\\]|\\^([^^\\s][^^]*?)\\^|~([^~\\s][^~]*?)~|(https?:\\/\\/[^\\s<]+))`, 'g');
const renderLink = (key, t, label, url) => (_jsx(Link, { url: url, children: _jsx(Text, { color: t.color.amber, underline: true, children: label }) }, key));
const trimBareUrl = (value) => {
    const trimmed = value.replace(/[),.;:!?]+$/g, '');
    return {
        tail: value.slice(trimmed.length),
        url: trimmed
    };
};
const renderAutolink = (key, t, raw) => {
    const url = raw.startsWith('mailto:') ? raw : raw.includes('@') && !raw.startsWith('http') ? `mailto:${raw}` : raw;
    return (_jsx(Link, { url: url, children: _jsx(Text, { color: t.color.amber, underline: true, children: raw.replace(/^mailto:/, '') }) }, key));
};
const indentDepth = (indent) => Math.floor(indent.replace(/\t/g, '  ').length / 2);
const parseFence = (line) => {
    const m = line.match(FENCE_RE);
    if (!m) {
        return null;
    }
    return {
        char: m[1][0],
        lang: m[2].trim().toLowerCase(),
        len: m[1].length
    };
};
const isFenceClose = (line, fence) => {
    const end = line.match(/^\s*(`{3,}|~{3,})\s*$/);
    return Boolean(end && end[1][0] === fence.char && end[1].length >= fence.len);
};
const isMarkdownFence = (lang) => ['md', 'markdown'].includes(lang);
const splitTableRow = (row) => row
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(cell => cell.trim());
const isTableDivider = (row) => {
    const cells = splitTableRow(row);
    return cells.length > 1 && cells.every(cell => TABLE_DIVIDER_CELL_RE.test(cell));
};
const stripInlineMarkup = (value) => value
    .replace(/!\[(.*?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '[image: $1] $2')
    .replace(/\[(.+?)\]\(((?:[^\s()]|\([^\s()]*\))+?)\)/g, '$1')
    .replace(/<((?:https?:\/\/|mailto:)[^>\s]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>/g, '$1')
    .replace(/~~(.+?)~~/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/_(.+?)_/g, '$1')
    .replace(/==(.+?)==/g, '$1')
    .replace(/\[\^([^\]]+)\]/g, '[$1]')
    .replace(/\^([^^\s][^^]*?)\^/g, '^$1')
    .replace(/~([^~\s][^~]*?)~/g, '_$1');
const renderTable = (key, rows, t) => {
    const widths = rows[0].map((_, ci) => Math.max(...rows.map(r => stripInlineMarkup(r[ci] ?? '').length)));
    return (_jsx(Box, { flexDirection: "column", paddingLeft: 2, children: rows.map((row, ri) => (_jsx(Box, { children: widths.map((width, ci) => {
                const cell = row[ci] ?? '';
                const pad = ' '.repeat(Math.max(0, width - stripInlineMarkup(cell).length));
                return (_jsxs(Text, { color: ri === 0 ? t.color.amber : undefined, children: [_jsx(MdInline, { t: t, text: cell }), pad, ci < widths.length - 1 ? '  ' : ''] }, ci));
            }) }, ri))) }, key));
};
function MdInline({ t, text }) {
    const parts = [];
    let last = 0;
    for (const m of text.matchAll(INLINE_RE)) {
        const i = m.index ?? 0;
        if (i > last) {
            parts.push(_jsx(Text, { children: text.slice(last, i) }, parts.length));
        }
        if (m[2] && m[3]) {
            parts.push(_jsxs(Text, { color: t.color.dim, children: ["[image: ", m[2], "] ", m[3]] }, parts.length));
        }
        else if (m[4] && m[5]) {
            parts.push(renderLink(parts.length, t, m[4], m[5]));
        }
        else if (m[6]) {
            parts.push(renderAutolink(parts.length, t, m[6]));
        }
        else if (m[7]) {
            parts.push(_jsx(Text, { strikethrough: true, children: m[7] }, parts.length));
        }
        else if (m[8]) {
            parts.push(_jsx(Text, { color: t.color.amber, dimColor: true, children: m[8] }, parts.length));
        }
        else if (m[9] || m[10]) {
            parts.push(_jsx(Text, { bold: true, children: m[9] ?? m[10] }, parts.length));
        }
        else if (m[11] || m[12]) {
            parts.push(_jsx(Text, { italic: true, children: m[11] ?? m[12] }, parts.length));
        }
        else if (m[13]) {
            parts.push(_jsx(Text, { backgroundColor: t.color.diffAdded, color: t.color.diffAddedWord, children: m[13] }, parts.length));
        }
        else if (m[14]) {
            parts.push(_jsxs(Text, { color: t.color.dim, children: ["[", m[14], "]"] }, parts.length));
        }
        else if (m[15]) {
            parts.push(_jsxs(Text, { color: t.color.dim, children: ["^", m[15]] }, parts.length));
        }
        else if (m[16]) {
            parts.push(_jsxs(Text, { color: t.color.dim, children: ["_", m[16]] }, parts.length));
        }
        else if (m[17]) {
            const { tail, url } = trimBareUrl(m[17]);
            parts.push(renderAutolink(parts.length, t, url));
            if (tail) {
                parts.push(_jsx(Text, { children: tail }, parts.length));
            }
        }
        last = i + m[0].length;
    }
    if (last < text.length) {
        parts.push(_jsx(Text, { children: text.slice(last) }, parts.length));
    }
    return _jsx(Text, { children: parts.length ? parts : _jsx(Text, { children: text }) });
}
function MdImpl({ compact, t, text }) {
    const nodes = useMemo(() => {
        const lines = text.split('\n');
        const nodes = [];
        let i = 0;
        let prevKind = null;
        const gap = () => {
            if (nodes.length && prevKind !== 'blank') {
                nodes.push(_jsx(Text, { children: " " }, `gap-${nodes.length}`));
                prevKind = 'blank';
            }
        };
        const start = (kind) => {
            if (prevKind && prevKind !== 'blank' && prevKind !== kind) {
                gap();
            }
            prevKind = kind;
        };
        while (i < lines.length) {
            const line = lines[i];
            const key = nodes.length;
            if (compact && !line.trim()) {
                i++;
                continue;
            }
            if (!line.trim()) {
                gap();
                i++;
                continue;
            }
            const fence = parseFence(line);
            if (fence) {
                const block = [];
                const lang = fence.lang;
                for (i++; i < lines.length && !isFenceClose(lines[i], fence); i++) {
                    block.push(lines[i]);
                }
                if (i < lines.length) {
                    i++;
                }
                if (isMarkdownFence(lang)) {
                    start('paragraph');
                    nodes.push(_jsx(Md, { compact: compact, t: t, text: block.join('\n') }, key));
                    continue;
                }
                start('code');
                const isDiff = lang === 'diff';
                const highlighted = !isDiff && isHighlightable(lang);
                nodes.push(_jsxs(Box, { flexDirection: "column", paddingLeft: 2, children: [lang && !isDiff && _jsx(Text, { color: t.color.dim, children: '─ ' + lang }), block.map((l, j) => {
                            if (highlighted) {
                                return (_jsx(Text, { children: highlightLine(l, lang, t).map(([color, text], k) => color ? (_jsx(Text, { color: color, children: text }, k)) : (_jsx(Text, { children: text }, k))) }, j));
                            }
                            const add = isDiff && l.startsWith('+');
                            const del = isDiff && l.startsWith('-');
                            const hunk = isDiff && l.startsWith('@@');
                            return (_jsx(Text, { backgroundColor: add ? t.color.diffAdded : del ? t.color.diffRemoved : undefined, color: add ? t.color.diffAddedWord : del ? t.color.diffRemovedWord : hunk ? t.color.dim : undefined, dimColor: isDiff && !add && !del && !hunk && l.startsWith(' '), children: l }, j));
                        })] }, key));
                continue;
            }
            if (line.trim().startsWith('$$')) {
                start('code');
                const block = [];
                for (i++; i < lines.length; i++) {
                    if (lines[i].trim().startsWith('$$')) {
                        i++;
                        break;
                    }
                    block.push(lines[i]);
                }
                nodes.push(_jsxs(Box, { flexDirection: "column", paddingLeft: 2, children: [_jsx(Text, { color: t.color.dim, children: "\u2500 math" }), block.map((l, j) => (_jsx(Text, { color: t.color.amber, children: l }, j)))] }, key));
                continue;
            }
            const heading = line.match(HEADING_RE);
            if (heading) {
                start('heading');
                nodes.push(_jsx(Text, { bold: true, color: t.color.amber, children: heading[2] }, key));
                i++;
                continue;
            }
            if (i + 1 < lines.length && line.trim()) {
                const setext = lines[i + 1].match(/^\s{0,3}(=+|-+)\s*$/);
                if (setext) {
                    start('heading');
                    nodes.push(_jsx(Text, { bold: true, color: t.color.amber, children: line.trim() }, key));
                    i += 2;
                    continue;
                }
            }
            if (HR_RE.test(line)) {
                start('rule');
                nodes.push(_jsx(Text, { color: t.color.dim, children: '─'.repeat(36) }, key));
                i++;
                continue;
            }
            const footnote = line.match(FOOTNOTE_RE);
            if (footnote) {
                start('list');
                nodes.push(_jsxs(Text, { color: t.color.dim, children: ["[", footnote[1], "] ", _jsx(MdInline, { t: t, text: footnote[2] ?? '' })] }, key));
                i++;
                while (i < lines.length && /^\s{2,}\S/.test(lines[i])) {
                    nodes.push(_jsx(Box, { paddingLeft: 2, children: _jsx(Text, { color: t.color.dim, children: _jsx(MdInline, { t: t, text: lines[i].trim() }) }) }, `${key}-cont-${i}`));
                    i++;
                }
                continue;
            }
            if (i + 1 < lines.length && DEF_RE.test(lines[i + 1])) {
                start('list');
                nodes.push(_jsx(Text, { bold: true, children: line.trim() }, key));
                i++;
                while (i < lines.length) {
                    const def = lines[i].match(DEF_RE);
                    if (!def) {
                        break;
                    }
                    nodes.push(_jsxs(Text, { children: [_jsx(Text, { color: t.color.dim, children: " \u00B7 " }), _jsx(MdInline, { t: t, text: def[1] })] }, `${key}-def-${i}`));
                    i++;
                }
                continue;
            }
            const bullet = line.match(/^(\s*)[-+*]\s+(.*)$/);
            if (bullet) {
                start('list');
                const depth = indentDepth(bullet[1]);
                const task = bullet[2].match(/^\[( |x|X)\]\s+(.*)$/);
                const marker = task ? (task[1].toLowerCase() === 'x' ? '☑' : '☐') : '•';
                const body = task ? task[2] : bullet[2];
                nodes.push(_jsxs(Text, { children: [_jsxs(Text, { color: t.color.dim, children: [' '.repeat(depth * 2), marker, ' '] }), _jsx(MdInline, { t: t, text: body })] }, key));
                i++;
                continue;
            }
            const numbered = line.match(/^(\s*)(\d+)[.)]\s+(.*)$/);
            if (numbered) {
                start('list');
                const depth = indentDepth(numbered[1]);
                nodes.push(_jsxs(Text, { children: [_jsxs(Text, { color: t.color.dim, children: [' '.repeat(depth * 2), numbered[2], ".", ' '] }), _jsx(MdInline, { t: t, text: numbered[3] })] }, key));
                i++;
                continue;
            }
            if (/^\s*(?:>\s*)+/.test(line)) {
                start('quote');
                const quoteLines = [];
                while (i < lines.length && /^\s*(?:>\s*)+/.test(lines[i])) {
                    const raw = lines[i];
                    const prefix = raw.match(/^\s*(?:>\s*)+/)?.[0] ?? '';
                    quoteLines.push({
                        depth: (prefix.match(/>/g) ?? []).length,
                        text: raw.slice(prefix.length)
                    });
                    i++;
                }
                nodes.push(_jsx(Box, { flexDirection: "column", children: quoteLines.map((ql, qi) => (_jsxs(Text, { color: t.color.dim, children: [' '.repeat(Math.max(0, ql.depth - 1) * 2), '│ ', _jsx(MdInline, { t: t, text: ql.text })] }, qi))) }, key));
                continue;
            }
            if (line.includes('|') && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
                start('table');
                const tableRows = [];
                tableRows.push(splitTableRow(line));
                i += 2;
                while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
                    tableRows.push(splitTableRow(lines[i]));
                    i++;
                }
                nodes.push(renderTable(key, tableRows, t));
                continue;
            }
            if (/^<details\b/i.test(line) || /^<\/details>/i.test(line)) {
                i++;
                continue;
            }
            const summary = line.match(/^<summary>(.*?)<\/summary>$/i);
            if (summary) {
                start('paragraph');
                nodes.push(_jsxs(Text, { color: t.color.dim, children: ["\u25B6 ", summary[1]] }, key));
                i++;
                continue;
            }
            if (/^<\/?[^>]+>$/.test(line.trim())) {
                start('paragraph');
                nodes.push(_jsx(Text, { color: t.color.dim, children: line.trim() }, key));
                i++;
                continue;
            }
            if (line.includes('|') && line.trim().startsWith('|')) {
                start('table');
                const tableRows = [];
                while (i < lines.length && lines[i].trim().startsWith('|')) {
                    const row = lines[i].trim();
                    if (!/^[|\s:-]+$/.test(row)) {
                        tableRows.push(splitTableRow(row));
                    }
                    i++;
                }
                if (tableRows.length) {
                    nodes.push(renderTable(key, tableRows, t));
                }
                continue;
            }
            start('paragraph');
            nodes.push(_jsx(MdInline, { t: t, text: line }, key));
            i++;
        }
        return nodes;
    }, [compact, t, text]);
    return _jsx(Box, { flexDirection: "column", children: nodes });
}
export const Md = memo(MdImpl);
