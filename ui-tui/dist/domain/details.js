const MODES = ['hidden', 'collapsed', 'expanded'];
const THINKING_FALLBACK = {
    collapsed: 'collapsed',
    full: 'expanded',
    truncated: 'collapsed'
};
export const parseDetailsMode = (v) => {
    const s = typeof v === 'string' ? v.trim().toLowerCase() : '';
    return MODES.find(m => m === s) ?? null;
};
export const resolveDetailsMode = (d) => parseDetailsMode(d?.details_mode) ??
    THINKING_FALLBACK[String(d?.thinking_mode ?? '')
        .trim()
        .toLowerCase()] ??
    'collapsed';
export const nextDetailsMode = (m) => MODES[(MODES.indexOf(m) + 1) % MODES.length];
