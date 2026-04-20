export const asRpcResult = (value) => !value || typeof value !== 'object' || Array.isArray(value) ? null : value;
export const asCommandDispatch = (value) => {
    const o = asRpcResult(value);
    if (!o || typeof o.type !== 'string') {
        return null;
    }
    const t = o.type;
    if (t === 'exec' || t === 'plugin') {
        return { type: t, output: typeof o.output === 'string' ? o.output : undefined };
    }
    if (t === 'alias' && typeof o.target === 'string') {
        return { type: 'alias', target: o.target };
    }
    if (t === 'skill' && typeof o.name === 'string') {
        return { type: 'skill', name: o.name, message: typeof o.message === 'string' ? o.message : undefined };
    }
    if (t === 'send' && typeof o.message === 'string') {
        return { type: 'send', message: o.message };
    }
    return null;
};
export const rpcErrorMessage = (err) => err instanceof Error && err.message ? err.message : typeof err === 'string' && err.trim() ? err : 'request failed';
