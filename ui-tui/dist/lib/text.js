import { THINKING_COT_MAX } from '../config/limits.js';
const ESC = String.fromCharCode(27);
const ANSI_RE = new RegExp(`${ESC}\\[[0-9;]*m`, 'g');
const WS_RE = /\s+/g;
export const stripAnsi = (s) => s.replace(ANSI_RE, '');
export const hasAnsi = (s) => s.includes(`${ESC}[`) || s.includes(`${ESC}]`);
const renderEstimateLine = (line) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('|')) {
        return trimmed
            .split('|')
            .filter(Boolean)
            .map(cell => cell.trim())
            .join('  ');
    }
    return line
        .replace(/!\[(.*?)\]\(([^)\s]+)\)/g, '[image: $1]')
        .replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\*\*(.+?)\*\*/g, '$1')
        .replace(/__(.+?)__/g, '$1')
        .replace(/\*(.+?)\*/g, '$1')
        .replace(/_(.+?)_/g, '$1')
        .replace(/~~(.+?)~~/g, '$1')
        .replace(/==(.+?)==/g, '$1')
        .replace(/\[\^([^\]]+)\]/g, '[$1]')
        .replace(/^#{1,6}\s+/, '')
        .replace(/^\s*[-*+]\s+\[( |x|X)\]\s+/, (_m, checked) => `• [${checked.toLowerCase() === 'x' ? 'x' : ' '}] `)
        .replace(/^\s*[-*+]\s+/, '• ')
        .replace(/^\s*(\d+)\.\s+/, '$1. ')
        .replace(/^\s*(?:>\s*)+/, '│ ');
};
export const compactPreview = (s, max) => {
    const one = s.replace(WS_RE, ' ').trim();
    return !one ? '' : one.length > max ? one.slice(0, max - 1) + '…' : one;
};
export const estimateTokensRough = (text) => (!text ? 0 : (text.length + 3) >> 2);
export const edgePreview = (s, head = 16, tail = 28) => {
    const one = s.replace(WS_RE, ' ').trim().replace(/\]\]/g, '] ]');
    return !one
        ? ''
        : one.length <= head + tail + 4
            ? one
            : `${one.slice(0, head).trimEnd()}.. ${one.slice(-tail).trimStart()}`;
};
export const pasteTokenLabel = (text, lineCount) => {
    const preview = edgePreview(text);
    if (!preview) {
        return `[[ [${fmtK(lineCount)} lines] ]]`;
    }
    const [head = preview, tail = ''] = preview.split('.. ', 2);
    return tail
        ? `[[ ${head.trimEnd()}.. [${fmtK(lineCount)} lines] .. ${tail.trimStart()} ]]`
        : `[[ ${preview} [${fmtK(lineCount)} lines] ]]`;
};
export const thinkingPreview = (reasoning, mode, max = THINKING_COT_MAX) => {
    const raw = reasoning.trim();
    return !raw || mode === 'collapsed' ? '' : mode === 'full' ? raw : compactPreview(raw.replace(WS_RE, ' '), max);
};
export const stripTrailingPasteNewlines = (text) => (/[^\n]/.test(text) ? text.replace(/\n+$/, '') : text);
export const toolTrailLabel = (name) => name
    .split('_')
    .filter(Boolean)
    .map(p => p[0].toUpperCase() + p.slice(1))
    .join(' ') || name;
export const formatToolCall = (name, context = '') => {
    const label = toolTrailLabel(name);
    const preview = compactPreview(context, 64);
    return preview ? `${label}("${preview}")` : label;
};
export const buildToolTrailLine = (name, context, error, note) => {
    const detail = compactPreview(note ?? '', 72);
    return `${formatToolCall(name, context)}${detail ? ` :: ${detail}` : ''} ${error ? ' ✗' : ' ✓'}`;
};
export const isToolTrailResultLine = (line) => line.endsWith(' ✓') || line.endsWith(' ✗');
export const parseToolTrailResultLine = (line) => {
    if (!isToolTrailResultLine(line)) {
        return null;
    }
    const mark = line.endsWith(' ✗') ? '✗' : '✓';
    const body = line.slice(0, -2);
    const [call, detail] = body.split(' :: ', 2);
    if (detail != null) {
        return { call, detail, mark };
    }
    const legacy = body.indexOf(': ');
    if (legacy > 0) {
        return { call: body.slice(0, legacy), detail: body.slice(legacy + 2), mark };
    }
    return { call: body, detail: '', mark };
};
export const isTransientTrailLine = (line) => line.startsWith('drafting ') || line === 'analyzing tool output…';
export const sameToolTrailGroup = (label, entry) => entry === `${label} ✓` ||
    entry === `${label} ✗` ||
    entry.startsWith(`${label}(`) ||
    entry.startsWith(`${label} ::`) ||
    entry.startsWith(`${label}:`);
export const lastCotTrailIndex = (trail) => {
    for (let i = trail.length - 1; i >= 0; i--) {
        if (!isToolTrailResultLine(trail[i])) {
            return i;
        }
    }
    return -1;
};
export const estimateRows = (text, w, compact = false) => {
    let fence = null;
    let rows = 0;
    for (const raw of text.split('\n')) {
        const line = stripAnsi(raw);
        const maybeFence = line.match(/^\s*(`{3,}|~{3,})(.*)$/);
        if (maybeFence) {
            const marker = maybeFence[1];
            const lang = maybeFence[2].trim();
            if (!fence) {
                fence = { char: marker[0], len: marker.length };
                if (lang) {
                    rows += Math.ceil((`─ ${lang}`.length || 1) / w);
                }
            }
            else if (marker[0] === fence.char && marker.length >= fence.len) {
                fence = null;
            }
            continue;
        }
        const inCode = Boolean(fence);
        const trimmed = line.trim();
        if (!inCode && trimmed.startsWith('|') && /^[|\s:-]+$/.test(trimmed)) {
            continue;
        }
        const rendered = inCode ? line : renderEstimateLine(line);
        if (compact && !rendered.trim()) {
            continue;
        }
        rows += Math.ceil((rendered.length || 1) / w);
    }
    return Math.max(1, rows);
};
export const flat = (r) => Object.values(r).flat();
const COMPACT_NUMBER = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1, notation: 'compact' });
export const fmtK = (n) => COMPACT_NUMBER.format(n).replace(/[KMBT]$/, s => s.toLowerCase());
export const pick = (a) => a[Math.floor(Math.random() * a.length)];
export const isPasteBackedText = (text) => /\[\[paste:\d+(?:[^\n]*?)\]\]|\[paste #\d+ (?:attached|excerpt)(?:[^\n]*?)\]/.test(text);
