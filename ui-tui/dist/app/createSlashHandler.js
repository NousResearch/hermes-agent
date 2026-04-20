import { parseSlashCommand } from '../domain/slash.js';
import { asCommandDispatch, rpcErrorMessage } from '../lib/rpc.js';
import { findSlashCommand } from './slash/registry.js';
import { getUiState } from './uiStore.js';
export function createSlashHandler(ctx) {
    const { gw } = ctx.gateway;
    const { catalog } = ctx.local;
    const { page, send, sys } = ctx.transcript;
    const handler = (cmd) => {
        const flight = ++ctx.slashFlightRef.current;
        const ui = getUiState();
        const sid = ui.sid;
        const parsed = parseSlashCommand(cmd);
        const argTail = parsed.arg ? ` ${parsed.arg}` : '';
        const stale = () => flight !== ctx.slashFlightRef.current || getUiState().sid !== sid;
        const guarded = (fn) => (r) => {
            if (!stale() && r) {
                fn(r);
            }
        };
        const guardedErr = (e) => {
            if (!stale()) {
                sys(`error: ${rpcErrorMessage(e)}`);
            }
        };
        const runCtx = { ...ctx, flight, guarded, guardedErr, sid, stale, ui };
        const found = findSlashCommand(parsed.name);
        if (found) {
            found.run(parsed.arg, runCtx, cmd);
            return true;
        }
        if (catalog?.canon) {
            const needle = `/${parsed.name}`.toLowerCase();
            const matches = [
                ...new Set(Object.entries(catalog.canon)
                    .filter(([alias]) => alias.startsWith(needle))
                    .map(([, canon]) => canon))
            ];
            if (matches.length === 1 && matches[0].toLowerCase() !== needle) {
                return handler(`${matches[0]}${argTail}`);
            }
            if (matches.length > 1) {
                sys(`ambiguous command: ${matches.slice(0, 6).join(', ')}${matches.length > 6 ? ', …' : ''}`);
                return true;
            }
        }
        gw.request('slash.exec', { command: cmd.slice(1), session_id: sid })
            .then(r => {
            if (stale()) {
                return;
            }
            const body = r?.output || `/${parsed.name}: no output`;
            const text = r?.warning ? `warning: ${r.warning}\n${body}` : body;
            const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2;
            long ? page(text, parsed.name[0].toUpperCase() + parsed.name.slice(1)) : sys(text);
        })
            .catch(() => {
            gw.request('command.dispatch', { arg: parsed.arg, name: parsed.name, session_id: sid })
                .then((raw) => {
                if (stale()) {
                    return;
                }
                const d = asCommandDispatch(raw);
                if (!d) {
                    return sys('error: invalid response: command.dispatch');
                }
                if (d.type === 'exec' || d.type === 'plugin') {
                    return sys(d.output || '(no output)');
                }
                if (d.type === 'alias') {
                    return handler(`/${d.target}${argTail}`);
                }
                if (d.type === 'skill') {
                    sys(`⚡ loading skill: ${d.name}`);
                    return d.message?.trim() ? send(d.message) : sys(`/${parsed.name}: skill payload missing message`);
                }
                if (d.type === 'send') {
                    return d.message?.trim() ? send(d.message) : sys(`/${parsed.name}: empty message`);
                }
            })
                .catch(guardedErr);
        });
        return true;
    };
    return handler;
}
