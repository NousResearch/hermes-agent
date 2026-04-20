import { LONG_MSG } from '../config/limits.js';
import { buildToolTrailLine, fmtK } from '../lib/text.js';
export const introMsg = (info) => ({ info, kind: 'intro', role: 'system', text: '' });
export const imageTokenMeta = (info) => {
    const { width, height, token_estimate: t } = info ?? {};
    return [width && height ? `${width}x${height}` : '', (t ?? 0) > 0 ? `~${fmtK(t)} tok` : '']
        .filter(Boolean)
        .join(' · ');
};
export const userDisplay = (text) => {
    if (text.length <= LONG_MSG) {
        return text;
    }
    const first = text.split('\n')[0]?.trim() ?? '';
    const words = first.split(/\s+/).filter(Boolean);
    const prefix = (words.length > 1 ? words.slice(0, 4).join(' ') : first).slice(0, 80);
    return `${prefix || '(message)'} [long message]`;
};
export const toTranscriptMessages = (rows) => {
    if (!Array.isArray(rows)) {
        return [];
    }
    const out = [];
    let pending = [];
    for (const row of rows) {
        if (!row || typeof row !== 'object') {
            continue;
        }
        const { context, name, role, text } = row;
        if (role === 'tool') {
            pending.push(buildToolTrailLine(name ?? 'tool', context ?? ''));
            continue;
        }
        if (typeof text !== 'string' || !text.trim()) {
            continue;
        }
        if (role === 'assistant') {
            out.push({ role, text, ...(pending.length && { tools: pending }) });
            pending = [];
        }
        else if (role === 'user' || role === 'system') {
            out.push({ role, text });
            pending = [];
        }
    }
    return out;
};
export const fmtDuration = (ms) => {
    const t = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(t / 3600);
    const m = Math.floor((t % 3600) / 60);
    const s = t % 60;
    return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
};
